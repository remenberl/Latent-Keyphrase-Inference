# Latent Keyphrase Inference (LAKI)
## Publication

* Jialu Liu, Xiang Ren, Jingbo Shang, Taylor Cassidy, Clare Voss and Jiawei Han, "**[Representing Documents via Latent Keyphrase Inference](http://jialu.info/paper/www2016-liu.pdf)**”, Proc. of the 25th Int. Conf. on World Wide Web (WWW'16), Montreal, Canada, April 2016.

## Notes

The current implementation requires [SegPhrase](https://github.com/shangjingbo1226/SegPhrase) to extract domain keyphrases. It has been added under this repository as a submodule.

## Requirements

We will take Ubuntu for example.

* g++ 4.8
```
$ sudo apt-get install g++-4.8
```
* python 2.7
```
$ sudo apt-get install python
```
* scikit-learn
```
$ sudo apt-get install pip
$ sudo pip install sklearn
```
* nltk
```
$ sudo pip install nltk
```

## Build

LAKI can be easily built by Makefile in the terminal.

```
$ make
```

## Default Run

```
$ ./train_dblp.sh  #train a LAKI model using DBLP dataset.
$ ./test/test_inference #receives a string query and returns top ranked document keyphrases
```

## Parameters
All the parameters are located in train_dblp.sh

```
INPUT=data/AMiner-Paper.txt
```
INPUT refers to the input file of LAKI, can be downloaded from AMiner. For other datasets, please refer to the format of file indicated by RAW_TEXT (each single line indicates a document) and comment out line 25-28.

```
OMP_NUM_THREADS=4
```
Number of threads.

```
NUM_KEYPHRASES=40000
```
Number of domain keyphrases extracted by SegPhrase

```
MIN_PHRASE_SUPPORT=10
```
Number of occurrences for a valid domain keyphrase in the corpus.


####For other parameters regarding each individual module, please check the corresponding cpp files.





