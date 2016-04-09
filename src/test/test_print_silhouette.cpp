#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model/link.h"
#include "model/model.h"
#include "model/node.h"
#include "tools/easyloggingpp.h"
#include "tools/stringhelper.h"

using namespace std;

_INITIALIZE_EASYLOGGINGPP


int main(int argc, char** argv)
{
  string model_path = "tmp/dblp/model.init";
  string vocabulary_path = "tmp/dblp/vocabulary.txt";
  int keyphrase_start_id = 100000000;
  _START_EASYLOGGINGPP(argc, argv);

  Model init_model(model_path, keyphrase_start_id);
  model_path = "tmp/dblp/model.now";
  Model model(model_path, keyphrase_start_id);

  unordered_map<int, string> vocabulary;
  unordered_map<int, double> vocabulary_stat;
  unordered_map<string, int> reverse_vocabulary;
  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      vocabulary[stoi(elements[0])] = elements[1];
      if (stoi(elements[0]) < keyphrase_start_id) {
        reverse_vocabulary[elements[1]] = stoi(elements[0]);
      }
    }
    vocabulary_file.close();
  }

  while(true) {
    string input;
    cout << "Enter Content: ";
    getline(cin, input);
    map<Link*, double> m;
    map<Node*, double> p;
    if (reverse_vocabulary.find(input) != reverse_vocabulary.end()) {
      auto node = init_model.GetNode(reverse_vocabulary[input]);
      vector<string> print_vector;
      node->PrintChildren(vocabulary, &print_vector, 30);
      cout << "Initial Silhouette " << node->GetId() << ": (" << vocabulary[node->GetId()] << ", " << node->GetLeakWeight() << ") " << endl;
      for (string line_to_print: print_vector) {
        cout << line_to_print << " ";
      }
      cout << endl;

      node = model.GetNode(reverse_vocabulary[input]);
      print_vector.clear();
      node->PrintChildren(vocabulary, &print_vector, 100);
      cout << "Silhouette " << node->GetId() << ": (" << vocabulary[node->GetId()] << ", " << node->GetLeakWeight() << ") " << endl;
      for (string line_to_print: print_vector) {
        cout << line_to_print << " ";
      }
      cout << endl;
      cout << endl;
    }
  }
}