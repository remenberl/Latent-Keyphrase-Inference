#ifndef __ENCODER__
#define __ENCODER__

#include <unordered_map>

#include "online/sparse_vector.h"

using namespace std;

class InferenceOnline;
class Model;
class SegPhraseParser;

class Encoder
{
  public:
    Encoder(string model_path, string segphrase_path, string keyphrase_path,
            int num_keyphrase,
            string vocabulary_path, int keyphrase_start_id,
            int max_running_time,
            int min_iter, int min_pruned_keyphrases, int max_pruned_keyphrases,
            double min_link_significance, int debug_output_freq);
    ~Encoder();
    void encode(string input, SparseVector* sparse_vector) const;
    Model* getModel();
    vector<string> segment(string text);

  private:
    SegPhraseParser* parser_;
    InferenceOnline* inference_;
    Model* model_;
    unordered_map<string, int> vocabulary_;
    unordered_map<int, double> weight_;
};

#endif
