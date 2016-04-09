#include <algorithm>
#include <sstream>
#include <vector>
#include <utility>

#include "model/node.h"
#include "model/link.h"

using namespace std;

Node::Node(int node_id, bool is_content_unit): id_(node_id), is_content_unit_(is_content_unit)
{
  if (!is_content_unit && node_id != 0) {
    is_keyphrase_ = true;
  } else {
    is_keyphrase_ = false;
  }
  leak_link_ = NULL;
}

bool Node::IsKeyphrase() {return is_keyphrase_;}

bool Node::IsContentUnit() {return is_content_unit_;}

int Node::GetId() const
{
  return id_;
}

double Node::GetLeakWeight() const
{
  return leak_weight_;
}

pair<Link*, double> Node::GetLeakLinkAndProb() const
{
  return make_pair(leak_link_, log_leak_activate_prob_);
}

Link* Node::GetLeakLink() const
{
  return leak_link_;
}

vector<tuple<Node*, Link*, double, double>> Node::GetChildren() const
{
  return children_;
}

vector<tuple<Node*, Link*, double, double>> Node::GetParents() const
{
  return parents_;
}

void Node::AddChildLink(Link* child_link)
{
  children_.push_back(make_tuple(child_link->GetChildNode(),
                                 child_link,
                                 child_link->GetWeight(),
                                 child_link->GetExpMWeight()));
}

void Node::AddParentLink(Link* parent_link)
{
  if (parent_link->GetParentNode()->GetId() == 0) {
  	leak_link_ = parent_link;
    leak_weight_ = leak_link_->GetWeight();
    log_leak_activate_prob_ = leak_link_->GetLog1MExpMWeight();
  } else {
    parents_.push_back(make_tuple(parent_link->GetParentNode(),
                                  parent_link,
                                  parent_link->GetWeight(),
                                  parent_link->GetExpMWeight()));
  }
}

void Node::ReloadLinks()
{
  for (auto& child: children_) {
    get<2>(child) = get<1>(child)->GetWeight();
    get<3>(child) = get<1>(child)->GetExpMWeight();
  }
  for (auto& parent: parents_) {
    get<2>(parent) = get<1>(parent)->GetWeight();
    get<3>(parent) = get<1>(parent)->GetExpMWeight();
  }
  if (leak_link_ != NULL) {
    leak_weight_ = leak_link_->GetWeight();
    log_leak_activate_prob_ = leak_link_->GetLog1MExpMWeight();
  }
}

unsigned int edit_distance(const std::string& s1, const std::string& s2)
{
  const std::size_t len1 = s1.size(), len2 = s2.size();
  std::vector<std::vector<unsigned int>> d(len1 + 1, std::vector<unsigned int>(len2 + 1));

  d[0][0] = 0;
  for(unsigned int i = 1; i <= len1; ++i) d[i][0] = i;
  for(unsigned int i = 1; i <= len2; ++i) d[0][i] = i;

  for(unsigned int i = 1; i <= len1; ++i)
    for(unsigned int j = 1; j <= len2; ++j)
      // note that std::min({arg1, arg2, arg3}) works only in C++11,
      // for C++98 use std::min(std::min(arg1, arg2), arg3)
     d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (s1[i - 1] == s2[j - 1] ? 0 : 1) });
  return d[len1][len2];
}

void Node::PrintChildren(const unordered_map<int, string>& vocabulary,
                         vector<string>* print_vector,
                         int children_size=3) const
{
  vector<pair<Link*, double>> link_weight;
	for (const auto& child: children_) {
    link_weight.push_back(make_pair(get<1>(child), get<2>(child)));
  }

  auto SortLinksByWeightDesc = [](const pair<Link*, double>& link1,
                                  const pair<Link*, double>& link2) {
    return link1.second > link2.second;
  };
  sort(link_weight.begin(), link_weight.end(), SortLinksByWeightDesc);

  vector<string> pool;

  int count = children_size;
  for (const auto& child_link: link_weight) {
    if (count <= 0) return;
    auto child_node = child_link.first->GetChildNode();
    if (child_node->IsContentUnit()) {
      string content_unit = vocabulary.at(child_node->GetId());
      unsigned int min_distance = 100;
      for (const auto& unit: pool) {
        unsigned int distance = edit_distance(unit, content_unit);
        if (distance < min_distance) {
          min_distance = distance;
        }
      }
      if (min_distance > 4) {
        std::ostringstream s;
        if (count == children_size) {
          s << content_unit << ": " << child_link.second;
        } else {
          s << content_unit << ": " << child_link.second;
        }
        print_vector->push_back(s.str());
        pool.push_back(content_unit);
        --count;
      }
    } /*else {
      // auto descendents = child_node->GetChildNode();
      // vector<string> sub_print_vector;
      // child_node->PrintChildren(vocabulary, &sub_print_vector, 1);
      // for (const string& element: sub_print_vector) {
      //   s << element;
      // }
      std::ostringstream s;
      s << "[" << vocabulary.at(child_node->GetId()) << "]: "
        << child_link.second << ", ";
      print_vector->push_back(s.str());
    }*/
    // --children_size;
  }
  // print_vector->push_back(" ...");
}