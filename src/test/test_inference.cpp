#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "online/encoder.h"
#include "model/model.h"
#include "model/node.h"
#include "tools/easyloggingpp.h"
#include "tools/stringhelper.h"

using namespace std;

_INITIALIZE_EASYLOGGINGPP

bool SortByEnergyDesc(const pair<int, double>& pair1, const pair<int, double>& pair2)
{
  return pair1.second > pair2.second;
}

int main(int argc, char** argv)
{
  string vocabulary_path = "tmp/dblp/vocabulary.txt";
  string model_path = "tmp/dblp/model.now.iter3";
  Model model(model_path, 100000000);

  _START_EASYLOGGINGPP(argc, argv);

  unordered_map<int, string> vocabulary;
  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      vocabulary[stoi(elements[0])] = elements[1];
    }
    vocabulary_file.close();
  }

  string segphrase_path = "tmp/dblp/segmentation.model";
  string keyphrase_path = "tmp/dblp/keyphrases.csv";
  Encoder encoder(model_path, segphrase_path, keyphrase_path, 50000,
                  vocabulary_path, 100000000, 30, 2500, 10, 100, 1e-5, 1);
  int count = 1000;
  while(count > 0) {
    string input;
    cout << "Enter Content: ";
    getline(cin, input);
    // input = "We describe an open-source toolkit for statistical machine translation whose novel contributions are (a) support for linguistically motivated factors, (b) confusion network decoding, and (c) efficient data formats for translation models and language models. In addition to the SMT decoder, the toolkit also includes a wide variety of tools for training, tuning and applying the system to many translation tasks.";
    SparseVector sparse_vector;
    encoder.encode(input, &sparse_vector);

    vector<pair<int, double>> node_energy;
    for (auto node: sparse_vector.values) {
      node_energy.push_back(make_pair(node.index, node.value));
    }
    sort(node_energy.begin(), node_energy.end(), SortByEnergyDesc);

    for (auto node: node_energy) {
      vector<string> print_vector;
      model.GetNode(node.first)->PrintChildren(vocabulary, &print_vector, 7);
      if (vocabulary.find(node.first) != vocabulary.end()) {
        cout << "Cluster " << node.first << ": (" << vocabulary[node.first] << ", " << node.second <<  ", " << model.GetNode(node.first)->GetLeakWeight()<< ") ";
      } else {
        cout << "Cluster " << node.first << ": (" << node.second << ") ";
      }
      for (string line_to_print: print_vector) {
        cout << line_to_print.substr(0, 50) << " ";
      }
      cout << endl;
    }
    cout << endl;
    --count;
  }
}