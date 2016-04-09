#include <cmath>

#include "model/link.h"
#include "model/node.h"
#include "tools/math.h"

Link::Link(Node* parent_node, Node* child_node, double weight)
{
  parent_node_ = parent_node;
  parent_id_ = parent_node->GetId();
  child_node_ = child_node;
  child_id_ = child_node->GetId();
  weight_ = weight;
  log_1_m_exp_m_weight_ = log1mexp(weight);
  exp_m_weight_ = exp(-weight);
}

double Link::GetWeight() const {return weight_;}

double Link::GetLog1MExpMWeight() const {return log_1_m_exp_m_weight_;}

double Link::GetExpMWeight() const {return exp_m_weight_;}

Node* Link::GetParentNode()
{
  return parent_node_;
}

Node* Link::GetChildNode()
{
  return child_node_;
}

void Link::SetWeight(double weight)
{
  weight_ = weight;
  log_1_m_exp_m_weight_ = log1mexp(weight);
  exp_m_weight_ = exp(-weight);
}