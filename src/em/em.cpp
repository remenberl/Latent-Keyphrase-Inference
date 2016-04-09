#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <omp.h>
#include <string>
#include <vector>

#include "em.h"
#include "inference/inference_em.h"
#include "model/link.h"
#include "model/model.h"
#include "model/node.h"
#include "tools/color.h"
#include "tools/easyloggingpp.h"
#include "tools/math.h"
#include "tools/stringhelper.h"

using namespace std;

EM::EM(Model* model, string data_file, double sample_training_ratio,
       int num_iterations, int batch_size, int num_thread,
       string vocabulary_path, string related_keyphrases_path,
       int keyphrase_start_id, int infer_print_freq, int infer_time,
       int min_pruned_keyphrases, int max_pruned_keyphrases, int min_infer_iter,
       double min_link_significance):
sample_training_ratio_(sample_training_ratio), loglikelihood_(0),
batch_size_(batch_size), iterations_(num_iterations), trained_num_(0),
model_(model), data_file_(data_file)
{
  omp_set_num_threads(num_thread);
  for (int i = 0; i < omp_get_max_threads(); i++) {
    inference_copies_.push_back(
      new InferenceEM(model, vocabulary_path, related_keyphrases_path,
                      keyphrase_start_id, infer_time,
                      min_infer_iter, min_pruned_keyphrases, max_pruned_keyphrases,
                      min_link_significance, infer_print_freq));
  }
  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      weight_[stoi(elements[0])] = stod(elements[2]);
    }
    vocabulary_file.close();
  }
}

EM::~EM()
{
  for (int i = 0; i < omp_get_max_threads(); i++) {
    delete inference_copies_[i];
  }
}

void EM::Train(string output_model_path) {
  LOG(INFO) << "EM algorithm has begun.";
  int iter = 0;
  while (iter < iterations_) {
    iter += 1;
    LOG(INFO) << YELLOW << "Start Iter." << iter << " E-step." << RESET;
    NextIteration(iter);
    SaveAndReset();
    // if (iter % 4 == 0 && iter > 8) {
    //   RemoveLink();
    // }
    model_->Dump(output_model_path + ".iter" + to_string(iter));
  }
  LOG(INFO) << "EM algorithm has finished.";
}

void EM::NextIteration(int iteration)
{
  ifstream training_data(data_file_);

  trained_num_ = 0;
  unordered_map<Link*, double>* m_copy = new unordered_map<Link*, double>[omp_get_max_threads()];
  unordered_map<Node*, double>* p_copy = new unordered_map<Node*, double>[omp_get_max_threads()];
  double* loglikelihood = new double[omp_get_max_threads()]();

  if (training_data.is_open()) {
    string new_line;
    bool file_load_complete = false;
    int chunk_size = batch_size_ / omp_get_max_threads() / 10;
    while (true) {
      vector<string> pool;
      for (int i = 0; i < batch_size_; i++)
      {
        if (getline(training_data, new_line)) {
          if (sample_training_ratio_ < 1 && (double)rand() / RAND_MAX > sample_training_ratio_) {
            continue;
          }
          pool.push_back(new_line);
        } else {
          file_load_complete = true;
          break;
        }
      }

      #pragma omp parallel for schedule(dynamic, chunk_size)
      for (unsigned int i = 0; i < pool.size(); i++) {
        map<int, double> nodes;
        map<Link*, double> tmp_m;
        map<Node*, double> tmp_p;

        string line = pool[i];
        vector<string> node_ids = split(line, ' ');
        for (auto& node_id: node_ids) {
          nodes[stoi(node_id)] += 1;
        }
        double norm = 0.0000001;
        // idf
        for (auto& unit : nodes) {
          unit.second *= 1 / (log(weight_.at(unit.first) + 1) + 1);
          norm = max(unit.second, norm);
        }
        for (auto& unit : nodes) {
          unit.second /= norm;
        }

        loglikelihood[omp_get_thread_num()] +=
            inference_copies_[omp_get_thread_num()]->DoInference(nodes, &tmp_m, &tmp_p, true);

        set<Node*> valid_nodes;
        double threshold = 0;

        for (auto key_val: tmp_p) {
          if (key_val.second > threshold ) {
            valid_nodes.insert(key_val.first);
            if (key_val.first->GetId() == 0 || key_val.first->GetId() >= 100000000 || key_val.second > 1)
              LOG(INFO) << key_val.first->GetId();
          }
          else
            continue;
          if (p_copy[omp_get_thread_num()].find(key_val.first) != p_copy[omp_get_thread_num()].end()) {
            p_copy[omp_get_thread_num()][key_val.first] += key_val.second;
          }
          else {
            p_copy[omp_get_thread_num()][key_val.first] = key_val.second;
          }
        }
        for (auto key_val: tmp_m) {
          if (valid_nodes.find(key_val.first->GetParentNode()) != valid_nodes.end() || key_val.first->GetParentNode()->GetId() == 0) {
            if (m_copy[omp_get_thread_num()].find(key_val.first) != m_copy[omp_get_thread_num()].end()) {
              m_copy[omp_get_thread_num()][key_val.first] += key_val.second;
            }
            else {
              m_copy[omp_get_thread_num()][key_val.first] = key_val.second;
            }
          }
        }
      }
      trained_num_ += pool.size();
      LOG(INFO) << "Finish doing inference for "
                << GREEN << trained_num_ << RESET
                << " sessions.";
      if (file_load_complete) {
        break;
      }
    }
    training_data.close();
  }

  LOG(INFO) << "Integrating m and p from different threads.";
  for (int i = 0; i < omp_get_max_threads(); i++) {
    for (auto key_val: m_copy[i]) {
      if (m_.find(key_val.first) != m_.end()) {
        m_[key_val.first] += key_val.second;
      }
      else {
        m_[key_val.first] = key_val.second;
      }
    }
    for (auto key_val: p_copy[i]) {
      if (p_.find(key_val.first) != p_.end()) {
        p_[key_val.first] += key_val.second;
      }
      else {
        p_[key_val.first] = key_val.second;
      }
    }
    loglikelihood_ += loglikelihood[i];
  }
  LOG(INFO) << "Integration completes. Loglikelihood: " << loglikelihood_;
  delete[] m_copy;
  delete[] p_copy;
}


void EM::SaveAndReset()
{
  auto links = model_->GetLinks();
  for (auto link: links) {
    if (m_.find(link) != m_.end()) {
      double link_weight = m_[link];
      auto parent_node = link->GetParentNode();
      double weight;
      if (parent_node->GetId() != 0) {
        weight = -log1p(-link_weight / p_[parent_node]);
      } else {
        weight = -log1p(-link_weight / trained_num_);
      }
      if (std::isnan(weight))
        LOG(INFO) << link_weight << " " << p_[parent_node];
      if (weight < 0) {
         LOG(INFO) << "Link weight smaller than 0.";
      }
      if (parent_node->GetId() == 0 && (weight <= 1e-8 || std::isnan(weight))) {
        link->SetWeight(1e-8);
      } else {
        if (weight > 1){
          weight = 1;
        }
        if (weight < 1e-8) {
          weight = 1e-8;
        }
        link->SetWeight(weight);
      }
    } else {
      link->SetWeight(1e-8);
    }
  }

  m_.clear();
  p_.clear();
  for (int i = 0; i < omp_get_max_threads(); i++) {
    inference_copies_[i]->ReloadModel();
  }
  loglikelihood_ = 0;
}
