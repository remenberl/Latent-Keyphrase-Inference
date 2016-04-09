#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>

#include "model/model.h"
#include "model/node.h"
#include "model/link.h"
#include "tools/color.h"
#include "tools/easyloggingpp.h"
#include "tools/inireader.h"
#include "tools/stringhelper.h"

using namespace std;

string ConnectPath(string folder, string file)
{
  if (folder[folder.size() - 1] == '/') {
    return folder + file;
  } else {
    return folder + '/' + file;
  }
}

Model::Model(string model_file_path, int keyphrase_start_id):
num_content_units_(0), num_keyphrases_(0), num_nodes_(0), num_links_(0)
{
  VLOG(3) << "Model Loading Starts.";
  ifstream model_file(model_file_path);
  VLOG(3) << "Loading from file " << model_file_path;
  if (model_file.is_open()) {
    string line;
    getline(model_file,line);

    while (getline(model_file, line)) {
      vector<string> elements = split(line, '\t');
      int parent_id = stoi(elements[0]);
      int child_id = stoi(elements[1]);
      double weight = stod(elements[2]);

      Node* parent_node;
      Node* child_node;
      if (id_to_node_.find(parent_id) != id_to_node_.end()) {
        parent_node = id_to_node_[parent_id];
      } else {
        parent_node = new Node(parent_id, false);
        nodes_.push_back(parent_node);
        id_to_node_[parent_id] = parent_node;
        if (parent_id != 0) {
          keyphrases_.push_back(parent_node);
          num_keyphrases_ += 1;
        }
        num_nodes_ += 1;
      }
      if (id_to_node_.find(child_id) != id_to_node_.end()) {
        child_node = id_to_node_[child_id];
      } else {
        child_node = new Node(child_id, child_id >= keyphrase_start_id);
        nodes_.push_back(child_node);
        id_to_node_[child_id] = child_node;
        if (child_id >= keyphrase_start_id) {
          content_units_.push_back(child_node);
          num_content_units_ += 1;
        } else {
          keyphrases_.push_back(child_node);
          num_keyphrases_ += 1;
        }
        num_nodes_ += 1;
      }
      if (weight <= 0 || std::isnan(weight)) {
        continue;
      }
      Link* link = new Link(parent_node, child_node, weight);
      links_.push_back(link);
      if (parent_id == 0) {
        leak_links_.push_back(link);
      }
      parent_node->AddChildLink(link);
      child_node->AddParentLink(link);
      num_links_ += 1;
    }
    model_file.close();
  }
  VLOG(3) << "Model Loading Finishes: " << GREEN << num_nodes_ <<" nodes, "
          << num_links_ << " links, "
          << num_content_units_ << " content units, "
          << num_keyphrases_ << " domain keyphrases." << RESET;

  assert(num_keyphrases_ + num_content_units_ + 1 == num_nodes_);

  auto SortKeyphrasesByIdDesc = [](Node* node1, Node* node2) {
    return node1->GetId() > node2->GetId();
  };

  // sort domain keyphrase nodes by decreasing ids for easier topological traversal
  sort(keyphrases_.begin(), keyphrases_.end(), SortKeyphrasesByIdDesc);
}

Model::~Model()
{
  for (auto node: nodes_) {
      delete node;
  }
  for (auto link: links_) {
      delete link;
  }
}

vector<Node*> Model::GetAllNodes() const {
  return nodes_;
}

vector<Node*> Model::GetContentUnits() const {
  return content_units_;
}

vector<Node*> Model::GetKeyphrases() const {
  return keyphrases_;
}

vector<Link*> Model::GetLinks() const {
  return links_;
}

vector<Link*> Model::GetLeakLinks() const {
  return leak_links_;
}

Node* Model::GetNode(int node_id) {
  if (id_to_node_.find(node_id) != id_to_node_.end()) return id_to_node_[node_id];
  return NULL;
}

void Model::Dump(string model_file_path)
{
  LOG(INFO) << "Model Saving Starts.";
  ofstream model_file(model_file_path);
  if (model_file.is_open()) {
    model_file << "parent\tchild\tweight\n";
    for (auto link: links_) {
      // if (link->GetWeight() > 0 && !std::isnan(link->GetWeight())) {
      model_file << *link->GetParentNode() << "\t"
                 << *link->GetChildNode() << "\t"
                 << link->GetWeight() << endl;
      // }
    }
    model_file.close();
  }
  LOG(INFO) << "Model Saving Finishes.";
}
