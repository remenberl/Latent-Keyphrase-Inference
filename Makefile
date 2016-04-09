export CC  = gcc
export CXX = g++-4.9
export CFLAGS = -std=c++11 -Wall -O3 -msse2 -fopenmp -I./src/

BIN = build/silhouetting build/batch_encode build/batch_parse build/batch_segment build/web_query
TEST = build/test_model build/test_inference build/test_print_silhouette build/test_parser

all: ${BIN} ${TEST}
test: ${TEST}

build/math.o: src/tools/math.h
	mkdir -p ./build
	$(CXX) $(CFLAGS) -c src/tools/math.cpp -o $@
build/stringhelper.o: src/tools/stringhelper.h
	$(CXX) $(CFLAGS) -c src/tools/stringhelper.cpp -o $@
build/link.o: src/model/link.cpp build/math.o
	$(CXX) $(CFLAGS) -c src/model/link.cpp -o $@
build/node.o: src/model/node.cpp
	$(CXX) $(CFLAGS) -c src/model/node.cpp -o $@
build/ini.o: src/tools/ini.c src/tools/inireader.cpp
	$(CXX) $(CFLAGS) -c src/tools/ini.c -o $@
	$(CXX) $(CFLAGS) -c src/tools/inireader.cpp -o build/inireader.o
build/model.o: src/model/model.cpp build/link.o build/node.o build/ini.o build/stringhelper.o
	$(CXX) $(CFLAGS) -c src/model/model.cpp -o $@
build/inference.o: src/inference/inference.cpp build/model.o
	$(CXX) $(CFLAGS) -c src/inference/inference.cpp -o $@
build/inference_em.o: src/inference/inference.cpp src/inference/inference_em.cpp build/model.o
	$(CXX) $(CFLAGS) -c src/inference/inference_em.cpp -o $@
build/inference_online.o: src/inference/inference.cpp src/inference/inference_online.cpp build/model.o
	$(CXX) $(CFLAGS) -c src/inference/inference_online.cpp -o $@
build/em.o: src/em/em.cpp build/model.o build/inference.o build/inference_em.o build/stringhelper.o
	$(CXX) $(CFLAGS) -c src/em/em.cpp -o $@
build/batch_encode: src/online/batch_encode.cpp src/online/encoder.cpp build/inference_online.o
	$(CXX) $(CFLAGS) src/online/batch_encode.cpp src/online/encoder.cpp build/*.o -o $@
build/batch_parse: src/parser/batch_parse.cpp src/parser/segphrase_parser.h
	$(CXX) $(CFLAGS) src/parser/batch_parse.cpp build/*.o -o $@
build/batch_segment: src/parser/batch_segment.cpp src/parser/segphrase_parser.h
	$(CXX) $(CFLAGS) src/parser/batch_segment.cpp build/*.o -o $@
build/silhouetting: src/silhouetting.cpp build/model.o build/em.o
	$(CXX) $(CFLAGS) src/silhouetting.cpp build/*.o -o build/silhouetting
build/web_query: src/online/web_query.cpp src/online/encoder.cpp build/inference_online.o
	$(CXX) $(CFLAGS) src/online/web_query.cpp src/online/encoder.cpp build/*.o -o $@

build/test_model: src/test/test_model.cpp build/model.o
	$(CXX) $(CFLAGS) src/test/test_model.cpp build/*.o -o $@
build/test_inference: build/inference_online.o src/test/test_inference.cpp build/model.o
	$(CXX) $(CFLAGS) -pg src/test/test_inference.cpp src/online/encoder.cpp build/*.o -o $@
build/test_print_silhouette: src/test/test_print_silhouette.cpp build/model.o
	$(CXX) $(CFLAGS) src/test/test_print_silhouette.cpp build/*.o -o $@
build/test_parser: src/test/test_parser.cpp src/parser/segphrase_parser.h
	$(CXX) $(CFLAGS) src/test/test_parser.cpp build/ini.o build/inireader.o -o $@

clean:
	rm -rf build/*
	# rm -rf tmp/*