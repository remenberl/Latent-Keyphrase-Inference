#include <algorithm>
#include <cassert>
#include <ctime>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <utility>

#include "inference/inference.h"
#include "model/link.h"
#include "model/model.h"
#include "model/node.h"
#include "tools/color.h"
#include "tools/easyloggingpp.h"
#include "tools/math.h"

using namespace std;

Inference::Inference(Model* model, int max_running_time=60,
  int min_iter=1000, int min_pruned_keyphrases=10, int max_pruned_keyphrases=100,
  double min_link_significance=0.01, int debug_output_freq=1):
model_(model), initial_log_prob_(0), log_prob_(0),
running_time_limit_(max_running_time), min_iter_num_(min_iter),
min_pruned_keyphrases_(min_pruned_keyphrases),
max_pruned_keyphrases_(max_pruned_keyphrases),
link_prune_threshold_(min_link_significance),
debug_output_freq_(debug_output_freq)
{
  ReloadModel();
}

void Inference::ReloadModel()
{
  node_log_diff_leak_.clear();
  node_log_leak_true_.clear();
  initial_log_prob_ = 0;
  vector<Link*> links = model_->GetLeakLinks();
  for (const auto& link: links) {
    double weight = link->GetWeight();
    initial_log_prob_ -= weight;
    const auto& child_node = link->GetChildNode();
    node_log_leak_true_[child_node] = link->GetLog1MExpMWeight();
    node_log_diff_leak_[child_node] = node_log_leak_true_[child_node] + weight;
  }
  for (const auto& node : model_->GetAllNodes()) {
    node->ReloadLinks();
  }
  log_prob_ = initial_log_prob_;
}

void Inference::DetectContentUnits(const map<int, double>& query)
{
  observed_content_units_.clear();
  for (const auto& id2weight : query) {
    Node* node = model_->GetNode(id2weight.first);
    if (node != NULL && node->IsContentUnit()) {
      observed_content_units_[node] = id2weight.second;
    }
  }
}

void Inference::Reset()
{
  current_configuration_.clear();
  current_active_links_.clear();

  log_prob_ = initial_log_prob_;
  int size = pruned_keyphrases_.size() + observed_content_units_.size();
  keyphrase_log_diff_all_.clear();
  keyphrase_log_diff_all_.resize(size, 0);
  keyphrase_log_as_parent_true_.clear();
  keyphrase_log_as_parent_true_.resize(size * 10, 0);
  keyphrase_log_as_parent_false_.clear();
  keyphrase_log_as_parent_false_.resize(size * 10, 0);
  keyphrase_log_diff_as_child_.clear();
  keyphrase_log_diff_as_child_.resize(size, 0);
  keyphrase_flip_prob_.clear();
  keyphrase_flip_prob_.resize(size, 0);
  node_active_weight_as_child_.clear();
  node_active_weight_as_child_.resize(size, 0);
  node_log_as_child_true_.clear();
  node_log_as_child_true_.resize(size, 0);

  parents_.clear();
  children_.clear();
  topid2node_.clear();
  node2topid_.clear();
  toplink2link_.clear();
  link2toplink_.clear();

  int node_index = 0;
  int link_index = 0;
  for (const auto& content_unit2weight: observed_content_units_) {
    Node* content_unit = content_unit2weight.first;
    int content_unit_id = node_index;
    current_configuration_[content_unit_id] = content_unit;
    node2topid_[content_unit] = content_unit_id;
    topid2node_[content_unit_id] = content_unit;
    ++node_index;
    // initial log prob of the bayesian network with all keyphrase nodes off
    log_prob_ += content_unit->GetLeakWeight();
    log_prob_ += content_unit2weight.second * node_log_diff_leak_[content_unit];
    node_active_weight_as_child_[content_unit_id] = content_unit->GetLeakWeight();
    node_log_as_child_true_[content_unit_id] = content_unit->GetLeakLink()->GetLog1MExpMWeight();

    parents_[content_unit_id] = vector<tuple<int, Node*, int, Link*>>();
    for (const auto& node_link_tuple: content_unit->GetParents()) {
      const auto& parent_node = get<0>(node_link_tuple);
      const auto& link = get<1>(node_link_tuple);
      if (pruned_keyphrases_.find(parent_node) != pruned_keyphrases_.end()) {
        if (node2topid_.find(parent_node) == node2topid_.end()) {
          node2topid_[parent_node] = node_index;
          topid2node_[node_index] = parent_node;
          ++node_index;
        }
        if (link2toplink_.find(link) == link2toplink_.end()) {
          link2toplink_[link] = link_index;
          toplink2link_[link_index] = link;
          ++link_index;
        }
        parents_[content_unit_id].push_back(
            make_tuple(node2topid_[parent_node], parent_node, link2toplink_[link], link));
      }
    }
  }

  for (const auto& keyphrase_node: pruned_keyphrases_) {
    if (node2topid_.find(keyphrase_node) == node2topid_.end()) {
      node2topid_[keyphrase_node] = node_index;
      topid2node_[node_index] = keyphrase_node;
      ++node_index;
    }
    int keyphrase_id = node2topid_[keyphrase_node];
    children_[keyphrase_id] = vector<tuple<int, Node*, int, Link*>>();
    keyphrase_log_diff_all_[keyphrase_id] = 0;
    for (const auto& node_link_tuple: keyphrase_node->GetChildren()) {
      const auto& child_node = get<0>(node_link_tuple);
      const auto& link = get<1>(node_link_tuple);
      if (observed_content_units_.find(child_node) == observed_content_units_.end()) {
        keyphrase_log_diff_all_[keyphrase_id] -= link->GetWeight();
      } else {
        if (link2toplink_.find(link) == link2toplink_.end()) {
          link2toplink_[link] = link_index;
          toplink2link_[link_index] = link;
          ++link_index;
        }
        if ((int)keyphrase_log_as_parent_true_.size() < link_index) {
          keyphrase_log_as_parent_true_.resize(link_index * 2, 0);
          keyphrase_log_as_parent_false_.resize(link_index * 2, 0);
        }
        int link_id = link2toplink_[link];
        keyphrase_log_as_parent_true_[link_id] =
            log1mexp(node_active_weight_as_child_[node2topid_[child_node]] + link->GetWeight());
        assert(keyphrase_log_as_parent_true_[link_id] <= 0);
        keyphrase_log_as_parent_false_[link_id] =
            node_log_as_child_true_[node2topid_[child_node]];

        // compute once a keyphrase node get flipped, how will the log_prob_ changes
        keyphrase_log_diff_all_[keyphrase_id] +=
            observed_content_units_[child_node] * (keyphrase_log_as_parent_true_[link_id] -
                                              keyphrase_log_as_parent_false_[link_id]);
      }

      if (pruned_keyphrases_.find(child_node) != pruned_keyphrases_.end() ||
          observed_content_units_.find(child_node) != observed_content_units_.end()) {
        if (node2topid_.find(child_node) == node2topid_.end()) {
          node2topid_[child_node] = node_index;
          topid2node_[node_index] = child_node;
          ++node_index;
        }
        if (link2toplink_.find(link) == link2toplink_.end()) {
          link2toplink_[link] = link_index;
          toplink2link_[link_index] = link;
          ++link_index;
        }
        if ((int)keyphrase_log_as_parent_true_.size() < link_index) {
          keyphrase_log_as_parent_true_.resize(link_index * 2, 0);
          keyphrase_log_as_parent_false_.resize(link_index * 2, 0);
        }
        children_[keyphrase_id].push_back(
            make_tuple(node2topid_[child_node], child_node, link2toplink_[link], link));
      }
    }

    parents_[keyphrase_id] = vector<tuple<int, Node*, int, Link*>>();
    for (const auto& node_link_tuple: keyphrase_node->GetParents()) {
      const auto& parent_node = get<0>(node_link_tuple);
      const auto& link = get<1>(node_link_tuple);
      if (pruned_keyphrases_.find(parent_node) != pruned_keyphrases_.end()) {
        if (node2topid_.find(parent_node) == node2topid_.end()) {
          node2topid_[parent_node] = node_index;
          topid2node_[node_index] = parent_node;
          ++node_index;
        }
        if (link2toplink_.find(link) == link2toplink_.end()) {
          link2toplink_[link] = link_index;
          toplink2link_[link_index] = link;
          ++link_index;
        }
        if ((int)keyphrase_log_as_parent_true_.size() < link_index) {
          keyphrase_log_as_parent_true_.resize(link_index * 2, 0);
          keyphrase_log_as_parent_false_.resize(link_index * 2, 0);
        }
        parents_[keyphrase_id].push_back(
            make_tuple(node2topid_[parent_node], parent_node, link2toplink_[link], link));
      }
    }

    node_active_weight_as_child_[keyphrase_id] = keyphrase_node->GetLeakWeight();
    keyphrase_log_diff_as_child_[keyphrase_id] = node_log_diff_leak_[keyphrase_node];
    node_log_as_child_true_[keyphrase_id] = node_log_leak_true_[keyphrase_node];
    keyphrase_log_diff_all_[keyphrase_id] += node_log_diff_leak_[keyphrase_node];

    // update the sampling probability of the keyphrase node according to Gibbs samping rule
    double tmp = exp(keyphrase_log_diff_all_[keyphrase_id]);
    keyphrase_flip_prob_[keyphrase_id] = tmp / (1 + tmp);
  }
}

void Inference::UpdateKeyphraseFlipProb(int node_id)
{
  double tmp = exp(keyphrase_log_diff_all_[node_id]);
  keyphrase_flip_prob_[node_id] = tmp / (1 + tmp);
}

void Inference::Flip(const pair<int, Node*>& keyphrase_id_and_node)
{
  bool turn_on = false;
  const auto& keyphrase_id = keyphrase_id_and_node.first;
  // compute the new log_prob_ aftehr flipping the keyphrase node
  if (current_configuration_.find(keyphrase_id) == current_configuration_.end()) {
    current_configuration_[keyphrase_id] = keyphrase_id_and_node.second;
    log_prob_ += keyphrase_log_diff_all_[keyphrase_id];
    turn_on = true;
  } else {
    current_configuration_.erase(keyphrase_id);
    log_prob_ -= keyphrase_log_diff_all_[keyphrase_id];
  }
  // UpdateKeyphraseFlipProb(keyphrase_id);

  for (const auto& node_link_tuple: children_[keyphrase_id]) {
    const auto& child_id = get<0>(node_link_tuple);
    const auto& child_node = get<1>(node_link_tuple);
    const auto& link_id = get<2>(node_link_tuple);
    const auto& link = get<3>(node_link_tuple);
    if (turn_on) {
      node_active_weight_as_child_[child_id] += link->GetWeight();
    } else {
      node_active_weight_as_child_[child_id] -= link->GetWeight();
    }

    // update data structures related to flipped keyphrase's children
    if (child_node->IsKeyphrase()) {
      node_log_as_child_true_[child_id] = log1mexp(node_active_weight_as_child_[child_id]);
      assert(node_log_as_child_true_[child_id] <= 0);
      double log_diff_as_child_
          = node_log_as_child_true_[child_id] +
            node_active_weight_as_child_[child_id];
      keyphrase_log_diff_all_[child_id] -= keyphrase_log_diff_as_child_[child_id];
      keyphrase_log_diff_all_[child_id] += log_diff_as_child_;
      keyphrase_log_diff_as_child_[child_id] = log_diff_as_child_;
      UpdateKeyphraseFlipProb(child_id);
    }
    if (child_node->IsContentUnit()) {
      node_log_as_child_true_[child_id] = log1mexp(node_active_weight_as_child_[child_id]);
    }

    // update data structures related to flipped keyphrase's spouses (child's parent)
    if (current_configuration_.find(child_id) != current_configuration_.end()) {
      if (turn_on) {
        current_active_links_[link_id] = make_pair(link, link->GetLog1MExpMWeight());
      } else {
        current_active_links_.erase(link_id);
      }
      for (const auto &node_link_tuple: parents_[child_id]) {
        const auto& parent_id = get<0>(node_link_tuple);
        // const auto& parent_node = get<1>(node_link_tuple);
        const auto& parent_link_id = get<2>(node_link_tuple);
        const auto& parent_link = get<3>(node_link_tuple);
        double weight = 1;
        if (child_node->IsContentUnit()) {
          weight = observed_content_units_[child_node];
        }
        if (parent_id != keyphrase_id) {
          keyphrase_log_diff_all_[parent_id] -=
              weight * (keyphrase_log_as_parent_true_[parent_link_id] -
                        keyphrase_log_as_parent_false_[parent_link_id]);
          if (current_configuration_.find(parent_id) == current_configuration_.end()) {
            keyphrase_log_as_parent_true_[parent_link_id] =
                log1mexp(node_active_weight_as_child_[child_id] + parent_link->GetWeight());
            assert(keyphrase_log_as_parent_true_[parent_link_id] <= 0);
            keyphrase_log_as_parent_false_[parent_link_id] = node_log_as_child_true_[child_id];
          } else {
            keyphrase_log_as_parent_true_[parent_link_id] = node_log_as_child_true_[child_id];
            keyphrase_log_as_parent_false_[parent_link_id] =
                log1mexp(node_active_weight_as_child_[child_id] - parent_link->GetWeight());
            assert(keyphrase_log_as_parent_false_[parent_link_id] <= 0);
          }
          keyphrase_log_diff_all_[parent_id] +=
              weight * (keyphrase_log_as_parent_true_[parent_link_id] -
                        keyphrase_log_as_parent_false_[parent_link_id]);
          UpdateKeyphraseFlipProb(parent_id);
        }
      }
    }
  }

  // update data structures related to flipped keyphrase's parent
  for (const auto& node_link_tuple: parents_[keyphrase_id]) {
    const auto& parent_id = get<0>(node_link_tuple);
    // const auto& parent_node = get<1>(node_link_tuple);
    const auto& link_id = get<2>(node_link_tuple);
    const auto& link = get<3>(node_link_tuple);
    if (turn_on) {
      // note the sign
      keyphrase_log_diff_all_[parent_id] += link->GetWeight();
      if (current_configuration_.find(parent_id) != current_configuration_.end()) {
        keyphrase_log_as_parent_true_[link_id] = log1mexp(node_active_weight_as_child_[keyphrase_id]);
        assert(keyphrase_log_as_parent_true_[link_id] <= 0);
        keyphrase_log_as_parent_false_[link_id] = log1mexp(node_active_weight_as_child_[keyphrase_id] - link->GetWeight());
        assert(keyphrase_log_as_parent_false_[link_id] <= 0);
      } else {
        keyphrase_log_as_parent_true_[link_id] = log1mexp(node_active_weight_as_child_[keyphrase_id] + link->GetWeight());
        assert(keyphrase_log_as_parent_true_[link_id] <= 0);
        keyphrase_log_as_parent_false_[link_id] = log1mexp(node_active_weight_as_child_[keyphrase_id]);
        assert(keyphrase_log_as_parent_false_[link_id] <= 0);
      }
      keyphrase_log_diff_all_[parent_id] += keyphrase_log_as_parent_true_[link_id] -
                                        keyphrase_log_as_parent_false_[link_id];
    } else {
      keyphrase_log_diff_all_[parent_id] -= keyphrase_log_as_parent_true_[link_id] -
                                        keyphrase_log_as_parent_false_[link_id];
      // note the sign
      keyphrase_log_diff_all_[parent_id] -= link->GetWeight();
    }

    if (current_configuration_.find(parent_id) != current_configuration_.end()) {
      if (turn_on) {
        current_active_links_[link_id] = make_pair(link, link->GetLog1MExpMWeight());
      } else {
        current_active_links_.erase(link_id);
      }
    }
    UpdateKeyphraseFlipProb(parent_id);
  }
}

bool Inference::SampleKeyphraseToFlip(int node_id)
{
  bool activated = ((double)rand() / RAND_MAX)  < keyphrase_flip_prob_[node_id];
  if (activated !=
      (current_configuration_.find(node_id) != current_configuration_.end()))
    return true;
  else
    return false;
}

