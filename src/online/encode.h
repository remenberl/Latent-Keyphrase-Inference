#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Node;
class Link;

class Encode
{
  public:
    Encode();
    ~Encode();
    SparseVector GetAllNodes() const;
    vector<Node*> GetConceptNodes() const;
    vector<Node*> GetTopicNodes() const;
    vector<Link*> GetLinks() const;
    vector<Link*> GetLeakLinks() const;
    void Dump(string model_folder);
    Node* GetNode(int node_id);

  private:
    vector<Node*> nodes_;
    vector<Node*> topic_nodes_;
    vector<Node*> concept_nodes_;
    vector<Link*> links_;
    vector<Link*> leak_links_;
    unordered_map<int, Node*> id_to_node_;
    int num_concepts_;
    int num_topics_;
    int num_nodes_;
    int num_links_;
};