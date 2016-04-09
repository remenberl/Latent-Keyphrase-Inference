#include <map>
#include <set>
#include <unordered_map>
#include <utility>

#include "inference/inference.h"

using namespace std;

class Link;
class Model;
class Node;

class InferenceEM: public Inference
{
  public:
    InferenceEM(Model* model, string vocabulary_path,
                string related_nodes_path, int keyphrase_start_id, int max_running_time,
                int min_iter, int min_pruned_keyphrases, int max_pruned_keyphrases,
                double min_link_significance, int debug_output_freq);
    virtual ~InferenceEM() {};
    double DoInference(const map<int, double>& node_ids,
                       map<Link*, double>* m,
                       map<Node*, double>* p,
                       bool lock_observed);

  protected:
    // from content unit nodes reduce the search space of possible keyphrase nodes
    // the general idea is to greedily consider how much one keyphrase
    // (only one keyphrase is on, others off) can explain the observed
    // ontent units compared to leak terms. we consider each candidate keyphrase node
    // in reversed topological order from bottom to top, such that we are
    // able to collapse links from children to descendents
    void Prune();
    void Reset();

    // transform keyphrase id to valid keyphrase nodes
    void DetectKeyphrases(const map<int, double>& query);

    // compute necessary statsitics for em update
    void ComputeStatistics();

    // for computing complete log-likelihood
    map<map<int, Node*>, double> loglikelihood_;
    map<map<int, Node*>, int> support_;

    set<Node*> observed_keyphrases_;
    map<Link*, double> m_pool_;
    map<Node*, int> p_pool_;
    unordered_map<Node*, vector<Node*>> related_nodes_;
    unordered_map<Node*, double> discard_ratio_;
};

