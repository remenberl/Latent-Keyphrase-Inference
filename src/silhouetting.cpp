#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "model/model.h"
#include "model/node.h"
#include "model/link.h"
#include "em/em.h"
#include "tools/easyloggingpp.h"
#include "tools/inireader.h"
#include "tools/stringhelper.h"

#define _ELPP_THREAD_SAFE

using namespace std;

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
  if (strcmp(argv[argc - 1], "--help") == 0 || strcmp(argv[argc - 1], "-help") == 0) {
    cerr << "[usage] -model <model file> -em <em-data> "
         << "-ratio <sample training ratio> -iter <num of iterations> "
         << "-batch <batch size> "
         << "-thread <num of threads> "
         << "-output <output model file> "
         << "-vocab <vocabulary file> -candi <candidate content units file> "
         << "-infer_print <print frequency during inference> "
         << "-infer_time <maximum running time for inference> "
         << "-infer_min <minimum document keyphrases> "
         << "-infer_max <maximum document keyphrases> "
         << "-infer_iter <minimum iterations of sampling> "
         << "-infer_link <minimum link significance score for pruning> "
         << endl;
    return -1;
  }

  // meta parameters for em module
  string model_file, data_file, output;
  int num_em_iter = 10, batch_size = 30000;
  int keyphrase_start_id = 100000000;
  int num_thread = 4;
  double ratio = 1;

  // parameters for inference
  string vocabulary_file, candidate_content_units_file;
  int infer_print_freq = 500, inference_time = 60;
  int min_pruned_keyphrases = 10, max_pruned_keyphrases = 100;
  int min_infer_iter = 30;
  double min_link_significance = 0.01;

  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-model") == 0) {
      model_file = argv[i + 1];
    } else if (strcmp(argv[i], "-em") == 0) {
      data_file = argv[i + 1];
    } else if (strcmp(argv[i], "-ratio") == 0) {
      ratio = atof(argv[i + 1]);
    } else if (strcmp(argv[i], "-iter") == 0) {
      num_em_iter = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch_size = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-thread") == 0) {
      num_thread = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-output") == 0) {
      output = argv[i + 1];
    } else if (strcmp(argv[i], "-vocab") == 0) {
      vocabulary_file = argv[i + 1];
    } else if (strcmp(argv[i], "-candi") == 0) {
      candidate_content_units_file = argv[i + 1];
    } else if (strcmp(argv[i], "-infer_print") == 0) {
      infer_print_freq = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-infer_time") == 0) {
      inference_time = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-infer_min") == 0) {
      min_pruned_keyphrases = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-infer_max") == 0) {
      max_pruned_keyphrases = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-infer_iter") == 0) {
      min_infer_iter = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-infer_link") == 0) {
      min_link_significance = atof(argv[i + 1]);
    }
  }

  _START_EASYLOGGINGPP(argc, argv);
  Model model(model_file, keyphrase_start_id);
  EM em(&model, data_file, ratio, num_em_iter, batch_size, num_thread,
        vocabulary_file, candidate_content_units_file, keyphrase_start_id,
        infer_print_freq, inference_time, min_pruned_keyphrases, max_pruned_keyphrases,
        min_infer_iter, min_link_significance);
  em.Train(output);
}