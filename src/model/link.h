#include <algorithm>
#include <vector>

using namespace std;

class Node;

class Link
{
  public:
    Link(Node* parent_node, Node* child_node, double weight);
    ~Link() {};
    double GetWeight() const;
    double GetLog1MExpMWeight() const;
    double GetExpMWeight() const;
    Node* GetParentNode();
    Node* GetChildNode();
    void SetWeight(double weight);
    int GetParentId() const {return parent_id_;}
    int GetChildId() const {return child_id_;}

    bool operator==(const Link& link) const {
        return (parent_id_ == link.GetParentId() &&
                child_id_ == link.GetChildId());
    }

  private:
    Node* parent_node_;
    Node* child_node_;
    int parent_id_;
    int child_id_;
    double weight_;

    // log(1-exp(-weight))
    double log_1_m_exp_m_weight_;

    // exp(-weight)
    double exp_m_weight_;
};

namespace std{
    template<>
    class hash<Link> {
        public :
            size_t operator()(const Link &link) const { 
                return std::hash<int>()(link.GetParentId() + link.GetChildId());   
            }
    };
}