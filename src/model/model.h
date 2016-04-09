#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Node;
class Link;

class Model
{
  public:
    Model(string model_file_path, int keyphrase_start_id);
    ~Model();
    vector<Node*> GetAllNodes() const;
    vector<Node*> GetContentUnits() const;
    vector<Node*> GetKeyphrases() const;
    vector<Link*> GetLinks() const;
    vector<Link*> GetLeakLinks() const;
    void Dump(string model_file_path);
    Node* GetNode(int node_id);

  private:
    vector<Node*> nodes_;
    vector<Node*> keyphrases_;
    vector<Node*> content_units_;
    vector<Link*> links_;
    vector<Link*> leak_links_;
    unordered_map<int, Node*> id_to_node_;
    int num_content_units_;
    int num_keyphrases_;
    int num_nodes_;
    int num_links_;
};