#include <algorithm>
#include <cassert>
#include <ctime>
#include <cmath>
#include <map>
#include <omp.h>
#include <random>
#include <set>
#include <unordered_map>
#include <utility>

#include "inference/inference_em.h"
#include "model/link.h"
#include "model/model.h"
#include "model/node.h"
#include "tools/color.h"
#include "tools/easyloggingpp.h"
#include "tools/math.h"
#include "tools/stringhelper.h"

using namespace std;

InferenceEM::InferenceEM(Model* model, string vocabulary_path,
  string related_nodes_path, int keyphrase_start_id, int max_running_time=60,
  int min_iter=1000, int min_pruned_keyphrases=10, int max_pruned_keyphrases=100,
  double min_link_significance=0.01, int debug_output_freq=300):
Inference(model, max_running_time, min_iter, min_pruned_keyphrases, max_pruned_keyphrases,
          min_link_significance, debug_output_freq)
{
  unordered_map<string, int> vocabulary;
  unordered_map<string, double> freq;
  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      if (stoi(elements[0]) < keyphrase_start_id) {
        vocabulary[elements[1]] = stoi(elements[0]);
      } else {
        freq[elements[1]] = stof(elements[2]) + 0.000000001;
      }
    }
    vocabulary_file.close();
  }

  for (const auto& key_val: vocabulary) {
    discard_ratio_[model->GetNode(key_val.second)] = min(1.0, 1 - 0.05 * (freq[key_val.first] + 1));
  }
  ifstream candidate_file(related_nodes_path);
  if (candidate_file.is_open()) {
    string line;
    while (getline(candidate_file, line)) {
      vector<string> elements = split(line, '\t');
      if (elements.size() <= 1 || vocabulary.find(elements[0]) == vocabulary.end()) {
        continue;
      }
      related_nodes_[model_->GetNode(vocabulary[elements[0]])] = vector<Node*>();
      for (unsigned int i = 1; i < elements.size(); i++) {
        if (vocabulary.find(elements[i]) == vocabulary.end() ||
            model_->GetNode(vocabulary[elements[i]]) == NULL) {
            continue;
        }
        related_nodes_[model_->GetNode(vocabulary[elements[0]])].push_back(model_->GetNode(vocabulary[elements[i]]));
      }
    }
    candidate_file.close();
  }
}

void InferenceEM::DetectKeyphrases(const map<int, double>& query)
{
  observed_keyphrases_.clear();
  for (const auto& id2weight: query) {
    Node* node = model_->GetNode(id2weight.first);
    if (node != NULL && node->IsKeyphrase()) {
      observed_keyphrases_.insert(node);
    }
  }
  // assert(observed_keyphrases_.size() > 0);
}

void InferenceEM::Prune()
{
  pruned_keyphrases_.clear();
  set<Node*> keyphrase_nodes;
  for (const auto& keyphrase_node: observed_keyphrases_) {
    if ((double)rand()/(double)RAND_MAX >= discard_ratio_[keyphrase_node])
      pruned_keyphrases_.insert(keyphrase_node);
    keyphrase_nodes.insert(keyphrase_node);
  }

  for (const auto& keyphrase_node: observed_keyphrases_) {
    for (const auto& related_node: related_nodes_[keyphrase_node]) {
      keyphrase_nodes.insert(related_node);
    }
  }

  vector<Node*> new_keyphrase_nodes;

  for (const auto& keyphrase_node: keyphrase_nodes) {
    if ((double)rand()/(double)RAND_MAX >= discard_ratio_[keyphrase_node]) {
      new_keyphrase_nodes.push_back(keyphrase_node);
    }
  }

  auto SortNodeByIdDesc = [](Node* node1, Node* node2) {
    return node1->GetId() > node2->GetId();
  };

  // sort keyphrase nodes by decreasing ids for easier topological traversal
  sort(new_keyphrase_nodes.begin(), new_keyphrase_nodes.end(), SortNodeByIdDesc);

  unordered_map<Node*, map<Node*, double>> all_collapsed_links;
  vector<pair<Node*, double>> keyphrase_pool;

  // precompute the log prob of all content units explained by leak term
  double content_units_leak_part_prob = 0;
  for (const auto& content_unit_weight: observed_content_units_) {
    content_units_leak_part_prob += content_unit_weight.second * node_log_leak_true_[content_unit_weight.first];
  }

  for (const auto& keyphrase_node: new_keyphrase_nodes) {
    // keeps -weight from the current keyphrase node to observed content unit nodes
    map<Node*, double> current_collapsed_links;
    const auto& children = keyphrase_node->GetChildren();
    for (const auto& node_link_tuple: children) {
      const auto& child_node = get<0>(node_link_tuple);
      const auto& link_wegiht = get<2>(node_link_tuple);
      // if (link_wegiht < link_prune_threshold_) continue;
      if (all_collapsed_links.find(child_node) != all_collapsed_links.end()) {
        // if the child node is a keyphrase node connected to observed ontent unit nodes
        const auto& child_collapsed_links = all_collapsed_links[child_node];
        for (auto child_collapsed_link: child_collapsed_links) {
          // increase_prob keeps the value of
          // log(1 - (1-exp(-weight)) * (1 - exp(child_collapsed_link.second)))
          // this is the -weight of the collapsed link from current node to child node
          // double current_weight = current_collapsed_links[child_collapsed_link.first];
          // if (child_collapsed_link.second > current_weight ||
          //     -link_wegiht > current_weight) continue;
          double increase_prob =
            log(1 - (1-get<3>(node_link_tuple)) * (1-exp(child_collapsed_link.second)));
            current_collapsed_links[child_collapsed_link.first] += increase_prob;
          assert(current_collapsed_links[child_collapsed_link.first] <= 0);
        }
      } else if (observed_content_units_.find(child_node) != observed_content_units_.end()) {
          current_collapsed_links[child_node] -= observed_content_units_[child_node] * link_wegiht;
        assert(current_collapsed_links[child_node] <= 0);
      }
    }
    // remove links whose weight is not significant enough
    if (link_prune_threshold_ > 0) {
      for (auto it = current_collapsed_links.cbegin(); it != current_collapsed_links.cend();) {
        VLOG(5) << "Link energy: " << RED << -it->second / keyphrase_node->GetLeakWeight() << RESET;
        if (-it->second / keyphrase_node->GetLeakWeight() < link_prune_threshold_) {
          current_collapsed_links.erase(it++);
        }
        else {
          ++it;
        }
      }
    }

    // compute P(node|others)
    // double energy = exp(log_prob_on) / (exp(log_prob_on) + exp(log_prob_off));
    double energy = 0;
    // if (link_prune_threshold_ > 0) {
    if (false) {
      double log_prob_on = node_log_leak_true_[keyphrase_node] +
                           content_units_leak_part_prob;
      double log_prob_off = content_units_leak_part_prob - keyphrase_node->GetLeakWeight();;
      for (const auto& content_unit_node: current_collapsed_links) {
        log_prob_on += -node_log_leak_true_[content_unit_node.first] +
                       log1mexp(content_unit_node.first->GetLeakWeight() - content_unit_node.second);
      }
      energy = exp(log_prob_on - logsumexp(vector<double>{log_prob_on, log_prob_off}));
      if (energy > link_prune_threshold_) {
        if (current_collapsed_links.size() > 0) {
          keyphrase_pool.push_back(make_pair(keyphrase_node, energy));
          all_collapsed_links[keyphrase_node] = current_collapsed_links;
        }
      }
    } else {
      double log_prob_on = node_log_leak_true_[keyphrase_node] + content_units_leak_part_prob;
      for (const auto& content_unit_node: current_collapsed_links) {
        log_prob_on += -node_log_leak_true_[content_unit_node.first] +
                       log1mexp(content_unit_node.first->GetLeakWeight() - content_unit_node.second);
      }
      energy = log_prob_on;
      if (current_collapsed_links.size() > 0) {
        keyphrase_pool.push_back(make_pair(keyphrase_node, energy));
        all_collapsed_links[keyphrase_node] = current_collapsed_links;
      }
    }
  }

  auto sortKeyphraseByEnergyDesc = [](const pair<Node*, double>& pair1,
                                  const pair<Node*, double>& pair2) {
    return pair1.second > pair2.second;
  };

  if (max_pruned_keyphrases_ < keyphrase_pool.size()) {
    partial_sort(keyphrase_pool.begin(),
                 keyphrase_pool.begin() + max_pruned_keyphrases_,
                 keyphrase_pool.end(),
                 sortKeyphraseByEnergyDesc);
  } else {
    sort(keyphrase_pool.begin(), keyphrase_pool.end(), sortKeyphraseByEnergyDesc);
  }

  for (unsigned int i = 0;
      pruned_keyphrases_.size() < max_pruned_keyphrases_ && i < keyphrase_pool.size(); i++) {
    if ((double)rand()/(double)RAND_MAX >= discard_ratio_[keyphrase_pool[i].first] &&
        observed_keyphrases_.find(keyphrase_pool[i].first) == observed_keyphrases_.end()) {
      pruned_keyphrases_.insert(keyphrase_pool[i].first);
    }
  }

  VLOG_EVERY_N(debug_output_freq_, 3) << "After pruning, there are "
                                      << RED << pruned_keyphrases_.size() << RESET
                                      <<" keyphrase nodes to flip.";

  if (VLOG_IS_ON(4) && keyphrase_pool.size() > 0) {
    VLOG_EVERY_N(debug_output_freq_, 4) << "Maximum energy: "
                                        << RED << keyphrase_pool[0].second << RESET
                                        << ", minimum energy: "
                                        << RED << keyphrase_pool[keyphrase_pool.size() - 1].second << RESET;
  }
}


void InferenceEM::Reset()
{
  Inference::Reset();
  m_pool_.clear();
  p_pool_.clear();
  loglikelihood_.clear();
  support_.clear();
}

void InferenceEM::ComputeStatistics()
{
  // compute unnormalized m for normal links
  for (const auto& active_link: current_active_links_) {
    const auto& link_id = active_link.first;
    const auto& link = active_link.second.first;
    const auto& weight = active_link.second.second;
    m_pool_[link] += exp(weight - keyphrase_log_as_parent_true_[link_id]);
    assert(weight < keyphrase_log_as_parent_true_[link_id]);
  }

  for (const auto& active_node: current_configuration_) {
    // compute unnormalized m for leak links
    const auto& node_id = active_node.first;
    const auto& node = active_node.second;
    const auto& log_leak_prob = node->GetLeakLinkAndProb();

    m_pool_[log_leak_prob.first] += exp(log_leak_prob.second -
                                        node_log_as_child_true_[node_id]);

    // compute unnormalized p
    if (node->IsKeyphrase()) {
      ++p_pool_[node];
    }
  }

  loglikelihood_[current_configuration_] = log_prob_;
  support_[current_configuration_] += 1;
}

double InferenceEM::DoInference(const map<int, double>& node_ids,
                                map<Link*, double>* m,
                                map<Node*, double>* p,
                                bool lock_observed=true)
{
  double begin_time = omp_get_wtime();
  Inference::DetectContentUnits(node_ids);
  DetectKeyphrases(node_ids);
  if (observed_keyphrases_.size() == 0) {
    return 0;
  }
  Prune();
  double pruning_time = double(omp_get_wtime() - begin_time) * 1000;
  if (pruned_keyphrases_.size() <= 1) {
    return 0;
  }
  Reset();
  if (lock_observed) {
    for (const auto& keyphrase_node: observed_keyphrases_) {
      if (pruned_keyphrases_.find(keyphrase_node) != pruned_keyphrases_.end()) {
        Inference::Flip(make_pair(Inference::Node2Topid(keyphrase_node), keyphrase_node));
      }
    }
  }

  m->clear();
  p->clear();
  double lapse;
  unsigned int iter = 0;
  unsigned int burn_in = min(500, int(min_iter_num_ * 0.4));

  set<pair<int, Node*>> pruned_keyphrase_ids_and_nodes;
  for (const auto& keyphrase_node: pruned_keyphrases_) {
    pruned_keyphrase_ids_and_nodes.insert(
        make_pair(Inference::Node2Topid(keyphrase_node), keyphrase_node));
  }
  set<pair<int, Node*>> focused_keyphrases;
  while(true) {
    if (iter < burn_in) {
      for (const auto& active_node: current_configuration_) {
        if (active_node.second->IsKeyphrase()) {
          focused_keyphrases.insert(active_node);
        }
      }
    } else if (iter == burn_in) {
      pruned_keyphrase_ids_and_nodes.clear();
      for (const auto& element: focused_keyphrases) {
        pruned_keyphrase_ids_and_nodes.insert(element);
      }
    } else {
      ComputeStatistics();
    }
    ++iter;

    // check whether time has been consumed
    lapse = double(omp_get_wtime() - begin_time) * 1000;
    if (lapse >= running_time_limit_ && iter >= min_iter_num_) {
      break;
    }

    for (const auto& node_id_pair: pruned_keyphrase_ids_and_nodes) {
      if (lock_observed &&
          observed_keyphrases_.find(node_id_pair.second) == observed_keyphrases_.end() &&
          Inference::SampleKeyphraseToFlip(node_id_pair.first)) {
        Inference::Flip(node_id_pair);
      }
      if (!lock_observed && Inference::SampleKeyphraseToFlip(node_id_pair.first)) {
        Inference::Flip(node_id_pair);
      }
    }
  }

  for (const auto& key_val: m_pool_) {
    (*m)[key_val.first] = (double)(key_val.second) / iter;
  }
  for (const auto& key_val: p_pool_) {
    (*p)[key_val.first] = (double)(key_val.second) / iter;
  }

  VLOG_EVERY_N(debug_output_freq_, 3) << "Inference contains " << iter << " iterations, "
                                      << observed_content_units_.size() << " ob content units, "
                                      << observed_keyphrases_.size() << " ob keyphrases, "
                                      << m_pool_.size() << " links, "
                                      << p_pool_.size() << " nodes, "
                                      << "pruning time: " << pruning_time
                                      << ", total time: " << lapse;
  vector<double> loglikelihood;
  double mean = 0;
  for (const auto& configuration: loglikelihood_) {
    double q = support_[configuration.first] / double(iter);
    loglikelihood.push_back(configuration.second - log(q));
    mean += configuration.second - log(q);
  }
  mean /= loglikelihood.size();
  return mean;
}
