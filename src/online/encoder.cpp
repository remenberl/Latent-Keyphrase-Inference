#include "online/encoder.h"

#include <cmath>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "inference/inference_online.h"
#include "model/link.h"
#include "model/model.h"
#include "model/node.h"
#include "online/sparse_vector.h"
#include "parser/segphrase_parser.h"
#include "tools/stringhelper.h"


Encoder::Encoder(string model_path, string segphrase_path, string keyphrase_path,
                 int num_keyphrase,
                 string vocabulary_path, int keyphrase_start_id,
                 int max_running_time,
                 int min_iter, int min_pruned_keyphrases, int max_pruned_keyphrases,
                 double min_link_significance, int debug_output_freq)
{
  model_ = new Model(model_path, keyphrase_start_id);
  parser_ = new SegPhraseParser(segphrase_path, keyphrase_path, num_keyphrase);
  inference_ = new InferenceOnline(model_, max_running_time,
                                   min_iter, min_pruned_keyphrases, max_pruned_keyphrases,
                                   min_link_significance, debug_output_freq);

  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      if (stoi(elements[0]) >= keyphrase_start_id) {
        vocabulary_[elements[1]] = stoi(elements[0]);
        weight_[stoi(elements[0])] = stod(elements[2]);
      }
    }
    vocabulary_file.close();
  }
}

Encoder::~Encoder()
{
  delete model_;
  delete parser_;
  delete inference_;
}

vector<string> Encoder::segment(string text)
{
  return parser_->segment(text);
}

Model* Encoder::getModel()
{
  return model_;
}

void Encoder::encode(string input, SparseVector* sparse_vector) const
{
  sparse_vector->clear();
  vector<string> elements = parser_->segment(input);
  map<int, double> units;
  for (const auto& element: elements) {
    if (vocabulary_.find(element) != vocabulary_.end()) {
      units[vocabulary_.at(element)] += 1;
    }
  }

  if (elements.size() >= 2) {
    for (unsigned int j = 2; j < 4; ++j) {
      for (unsigned int i = 0; i < elements.size() - j + 1; ++i) {
        string combined_unit = elements[i];
        for (unsigned int k = 1; k < j; ++k) {
          combined_unit += " " + elements[i + k];
        }
        if (vocabulary_.find(combined_unit) != vocabulary_.end()) {
          units[vocabulary_.at(combined_unit)] += 1;
        }
      }
    }
  }

  double norm = 0.0000001;
  for (auto& unit : units) {
    unit.second *= 1 / (log(weight_.at(unit.first) + 1) + 1);
    norm = max(unit.second, norm);
  }
  for (auto& unit : units) {
    unit.second /= norm;
  }

  map<Node*, double> p;
  inference_->DoInference(units, &p);

  for (const auto& node: p) {
    if (node.second < 10e-10 || std::isnan(node.second)) {
      continue;
    }
    else if (node.second >= 0.1) {
      // sparse_vector->push_back(node.first->GetId(), 1);
      // cout << node.second << " ";
      sparse_vector->push_back(node.first->GetId(), 0.5 + (node.second - 0.1) * 5 / 9);
    }
    else {
      sparse_vector->push_back(node.first->GetId(), log(0.1) / log(node.second) / 2);
    }
    // else {
    //   sparse_vector->push_back(node.first->GetId(),  (10 + log(node.second) / log(10))*0.1 );
    // }
  }
  sparse_vector->sort();
  // sparse_vector->normalize();
}

