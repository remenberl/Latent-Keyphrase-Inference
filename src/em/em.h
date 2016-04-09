#include <map>
#include <string>
#include <unordered_map>

using namespace std;

class InferenceEM;
class Link;
class Model;
class Node;

class EM
{
  public:
    EM(Model* model, string data_file, double sample_training_ratio,
       int num_iterations, int batch_size, int num_thread,
       string vocabulary_path, string related_domain_keyphrase_path,
       int keyphrase_start_id, int infer_print_freq, int infer_time,
       int min_pruned_keyphrases, int max_pruned_keyphrases, int min_infer_iter,
       double min_link_significance);
    ~EM();
    void Train(string output_model_path);
    void NextIteration(int iteration);
    void RemoveLink();

  private:
    void SaveAndReset();

    // every em iteration will sample a certain amount of training data.
    double sample_training_ratio_;
    double loglikelihood_;
    // controling link removing if the link is not significant
    // double min_link_significance_ratio_;
    int batch_size_;
    int iterations_;
    int trained_num_;
    unordered_map<Link*, double> m_;
    unordered_map<Node*, double> p_;
    Model* model_;
    string data_file_;
    vector<InferenceEM*> inference_copies_;
    unordered_map<int, double> weight_;
};


