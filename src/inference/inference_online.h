#include <map>
#include <set>
#include <utility>

#include "inference/inference.h"

using namespace std;

class Link;
class Model;
class Node;

class InferenceOnline: public Inference
{
  public:
    InferenceOnline(Model* model, int max_running_time,
                    int min_iter, int min_doc_keyphrases, int max_doc_keyphrases,
                    double min_link_significance, int debug_output_freq);
    virtual ~InferenceOnline() {};
    void DoInference(const map<int, double>& node_ids, map<Node*, double>* p);

  protected:
    // reduce the search space of possible keyphrase nodes.
    // the general idea is to greedily consider how much one keyphrase
    // (only one keyphrase is on, others off) can explain the observed
    // concepts compared to leak terms. we consider each candidate keyphrase node
    // in reversed topological order from bottom to top, such that we are
    // able to collapse links from children to descendents
    void Prune();
    void Reset();

    // compute necessary statsitics for online inference
    void ComputeStatistics();

    unordered_map<Node*, int> p_pool_;

};

