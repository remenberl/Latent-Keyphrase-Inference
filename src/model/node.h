#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

class Link;

class Node
{
  public:
    Node(int node_id, bool is_content_unit);
    ~Node() {};
    bool IsContentUnit();
    bool IsKeyphrase();
    int GetId() const;
    double GetLeakWeight() const;
    pair<Link*, double> GetLeakLinkAndProb() const;
    Link* GetLeakLink() const;
    vector<tuple<Node*, Link*, double, double>> GetChildren() const;
    vector<tuple<Node*, Link*, double, double>> GetParents() const;
    void ReloadLinks();
    void AddChildLink(Link* child_link);
    void AddParentLink(Link* parent_link);
    void PrintChildren(const unordered_map<int, string>& vocabulary,
                       vector<string>* print_vector,
                       int children_size) const;
    friend ostream & operator<<(ostream & os, Node const & node) {
      return os << to_string(node.id_);
    }
    bool operator==(const Node& node) const {
        return id_ == node.GetId();
    }

  private:
    int id_;
    bool is_content_unit_;
    bool is_keyphrase_;
    Link* leak_link_;
    vector<tuple<Node*, Link*, double, double>> children_;
    vector<tuple<Node*, Link*, double, double>> parents_;
    double leak_weight_;
    double log_leak_activate_prob_;
};

namespace std{
    template<>
    class hash<Node> {
        public :
            size_t operator()(const Node &node) const {
                return std::hash<int>()(node.GetId());
            }
    };
}