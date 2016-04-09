#!/bin/sh
INPUT=data/AMiner-Paper.txt

export OMP_NUM_THREADS=4
export NUM_KEYPHRASES=40000
export MIN_PHRASE_SUPPORT=10

make

Green='\033[0;32m'
NC='\033[0m'
if [ $INPUT == "data/AMiner-Paper.txt" ] && [ ! -e data/AMiner-Paper.txt ]; then
	echo -e "${Green}Downloading dataset${NC}"
	mkdir -p data
	wget http://jialu.info/datasets/AMiner-Paper.txt.gz -O data/AMiner-Paper.txt.gz
	gzip -d data/AMiner-Paper.txt.gz -f
fi

export PYTHON=~/python-env/bin/python
export PYPY=python
if type "pypy" > /dev/null; then
	export PYPY=pypy
fi

export RAW_TEXT=tmp/dblp/pure_text.txt
echo -e "${Green}Preprocessing DBLP Input File${NC}"
mkdir -p tmp/dblp
${PYPY} src/preprocessing/dblp/prepare_data.py -input $INPUT -output $RAW_TEXT

echo -e "${Green}Training SegPhrase${NC}"
./domain_keyphrase_extraction.sh

mv SegPhrase/results/vectors.bin tmp/dblp/vectors.bin
mv SegPhrase/results/salient.csv tmp/dblp/keyphrases.csv
mv SegPhrase/results/segmentation.model tmp/dblp/segmentation.model
mv SegPhrase/results/w2w_nn.txt tmp/dblp/ann.txt

echo -e "${Green}Identifying Phrases in Input File${NC}"
./SegPhrase/bin/segphrase_parser tmp/dblp/segmentation.model \
  tmp/dblp/keyphrases.csv $NUM_KEYPHRASES $RAW_TEXT ./tmp/dblp/segmented_text.txt 1

echo -e "${Green}Generating EM Training Data${NC}"
# detect phrases not noun
${PYPY} src/tools/detect_noise.py -input tmp/dblp/keyphrases.csv -output tmp/dblp/noise.txt
${PYPY} src/initialization/generate_em_training_data.py \
      -input tmp/dblp/segmented_text.txt \
      -stopwords src/tools/stopwords.txt \
      -word2vec tmp/dblp/vectors.bin \
      -noise tmp/dblp/noise.txt \
      -window 100 \
      -merge 1 \
      -vocab tmp/dblp/vocabulary.txt \
      -em tmp/dblp/training_data.txt \
      -merged tmp/dblp/synonym.txt

echo -e "${Green}Generating Candidate Content Units${NC}"
${PYPY} src/initialization/generate_candidate_content_units.py \
      -em tmp/dblp/training_data.txt \
      -merged tmp/dblp/synonym.txt \
      -vocab tmp/dblp/vocabulary.txt \
      -word2vec tmp/dblp/vectors.bin \
      -ann tmp/dblp/ann.txt \
      -child 50 \
      -candi tmp/dblp/candidate_content_units.txt

echo -e "${Green}Initializing Bayesian Network Structure${NC}"
${PYPY} src/initialization/initialize_model.py \
      -em tmp/dblp/training_data.txt \
      -candi tmp/dblp/candidate_content_units.txt \
      -vocab tmp/dblp/vocabulary.txt \
      -word2vec tmp/dblp/vectors.bin \
      -model tmp/dblp/model.init

echo -e "${Green}Domain Keyphrase Silhouetting${NC}"
./build/silhouetting -model tmp/dblp/model.init \
                  -em tmp/dblp/training_data.txt \
                  -ratio 0.5 -iter 3 -batch ${NUM_KEYPHRASES} \
                  -thread ${OMP_NUM_THREADS} \
                  -output tmp/dblp/model.now \
                  -vocab tmp/dblp/vocabulary.txt \
                  -candi tmp/dblp/candidate_content_units.txt \
                  -infer_time 50 -infer_min 10 -infer_max 100 \
                  -infer_iter 500 -infer_link 1e-6 -infer_print 3000 \
                  # --v=3
