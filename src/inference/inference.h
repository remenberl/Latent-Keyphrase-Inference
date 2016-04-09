#include <map>
#include <set>
#include <unordered_map>
#include <utility>

using namespace std;

class Link;
class Model;
class Node;

class Inference
{
  public:
    Inference(Model* model, int max_running_time,
              int min_iter, int min_pruned_keyphrases, int max_pruned_keyphrases,
              double min_link_significance, int debug_output_freq);
    ~Inference() {};

    // when model has been changed (link weights changed, structure adjusted),
    // this function need to be called to recompute some variables related
    // to the model
    void ReloadModel();

  protected:
    // transform id to valid domain keyphrase nodes
    void DetectContentUnits(const map<int, double>& query);

    // reduce the search space of possible domain keyphrase nodes
    virtual void Prune()=0;

    // reset variables for new inference
    virtual void Reset();

    // check whether a keyphrase node needs to flip its value according to
    // keyphrase_flip_prob_ after sampling
    virtual bool SampleKeyphraseToFlip(int node_id);

    // update keyphrase_flip_prob_ after flip
    virtual void UpdateKeyphraseFlipProb(int node_id);

    // flip a keyphrase node and update corresponding log-diff statistics
    void Flip(const pair<int, Node*>& keyphrase_id_and_node);

    Node* Topid2Node(const int& id) {return topid2node_[id];}
    int Node2Topid(Node* node) {return node2topid_[node];}

    Model* model_;

    // initial log prob for all nodes to be false including content unit nodes
    double initial_log_prob_;

    // log prob for current configuration
    double log_prob_;

    // keep all observed content unit nodes and their tf-idf weights
    map<Node*, double> observed_content_units_;

    // maximum running time for each inference
    double running_time_limit_;

    unsigned int min_iter_num_;
    unsigned int min_pruned_keyphrases_;
    unsigned int max_pruned_keyphrases_;

    // threshold of link importance used in the Prune() function
    double link_prune_threshold_;

    // threshold of node importance used in the Prune() function
    double node_prune_threshold_;

    // used to output debug information every debug_output_freq_ inferences
    int debug_output_freq_;

    // current active nodes during the sampling process
    map<int, Node*> current_configuration_;

    // active links between true parents and children
    map<int, pair<Link*, double>> current_active_links_;

    // interested keyphrases after pruning
    set<Node*> pruned_keyphrases_;

    // vector storing flip probibilities of certain domain keyphrase nodes,
    // used in MCMC for sampling a keyphrase node to flip
    vector<double> keyphrase_flip_prob_;

    // log P(domain keyphrase node=true | other nodes) -
    // log P(domain keyphrase node=false | other nodes)
    // used for updating keyphrase_flip_prob_
    vector<double> keyphrase_log_diff_all_;

    // the following vectors store log probabilities of certain pairs
    // between parent domain keyphrase node j and child node k,
    // used for computing keyphrase_log_diff_all_ and updating m.
    // log P(child k=true | parent j=true, other parents)
    vector<double> keyphrase_log_as_parent_true_;
    // log P(child k=true | parent j=false, other parents)
    vector<double> keyphrase_log_as_parent_false_;

    // log P(domain keyphrase node=true | parents) - log P(domain keyphrase node=false | parents)
    // used for computing keyphrase_log_diff_all_
    vector<double> keyphrase_log_diff_as_child_;

    // log P(node=true | parents)
    // used for computing keyphrase_log_as_parent_false_, keyphrase_log_as_parent_true_
    // and updating m
    vector<double> node_log_as_child_true_;

    // weight summation of all active parents
    // used for quickly updating node_log_as_child_true_,
    vector<double> node_active_weight_as_child_;

    // log P(node=true | noise, all other parents false) -
    // log P(node=false | noise, all other parents false)
    // used for quickly resetting all nodes in the network
    unordered_map<Node*, double> node_log_diff_leak_;

    // log P(node=true | noise, all other parents false)
    // used for updating m for leak links
    unordered_map<Node*, double> node_log_leak_true_;

    // for quickly accessing markov blankets
    map<int, vector<tuple<int, Node*, int, Link*>>> parents_;
    map<int, vector<tuple<int, Node*, int, Link*>>> children_;
    map<int, Node*> topid2node_;
    map<Node*, int> node2topid_;
    map<int, Link*> toplink2link_;
    map<Link*, int> link2toplink_;
};

