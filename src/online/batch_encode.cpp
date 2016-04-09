#include <fstream>
#include <omp.h>
#include <string>

#include "online/encoder.h"
#include "online/sparse_vector.h"
#include "tools/easyloggingpp.h"

using namespace std;

#define _ELPP_THREAD_SAFE
_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
  if (strcmp(argv[argc - 1], "--help") == 0 || strcmp(argv[argc - 1], "-help") == 0) {
    cerr << "[usage] -input <input text snippets> "
         << "-batch <batch size> -thread <num of threads> "
         << "-output <output encoding file> "
         << "-model <model file> -segphrase <segphrase model file> "
         << "-keyphrase <keyphrase file> "
         << "-num <number of keyphrases> -vocab <vocabulary file> "
         << "-prefix <keyphrase start id> "
         << "-infer_print <print frequency during inference> "
         << "-infer_time <maximum running time for inference> "
         << "-infer_min <minimum size of keyphrases> "
         << "-infer_max <maximum size of keyphrases> "
         << "-infer_iter <minimum iterations of sampling> "
         << "-infer_link <minimum link significance score for pruning> "
         << endl;
    return -1;
  }


  // meta parameters for encoding module
  string input_file, output_file;
  int batch_size = 1000, num_thread = 4;

  // parameters for inference
  string model_file, segphrase_file, keyphrase_file, vocabulary_file;
  int num_keyphrases=50000;
  int prefix=100000000;
  int inference_time = 60, min_infer_iter = 3000;
  int min_pruned_keyphrases = 10, max_pruned_keyphrases=100;
  double min_link_significance = 0.01;
  int infer_print_freq = 1;

  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-input") == 0) {
      input_file = argv[i + 1];
    } else if (strcmp(argv[i], "-batch") == 0) {
      batch_size = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-thread") == 0) {
      num_thread = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-output") == 0) {
      output_file = argv[i + 1];
    } else if (strcmp(argv[i], "-model") == 0) {
      model_file = argv[i + 1];
    } else if (strcmp(argv[i], "-segphrase") == 0) {
      segphrase_file = argv[i + 1];
    } else if (strcmp(argv[i], "-keyphrase") == 0) {
      keyphrase_file = argv[i + 1];
    } else if (strcmp(argv[i], "-num") == 0) {
      num_keyphrases = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-vocab") == 0) {
      vocabulary_file = argv[i + 1];
    } else if (strcmp(argv[i], "-prefix") == 0) {
      prefix = atoi(argv[i + 1]);
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
  omp_set_num_threads(num_thread);
  vector<Encoder*> encoders(omp_get_max_threads());

  // #pragma omp parallel
  for (int i = 0; i < omp_get_max_threads(); i++) {
    encoders[i] = new Encoder(model_file, segphrase_file, keyphrase_file,
                              num_keyphrases, vocabulary_file, prefix,
                              inference_time, min_infer_iter,
                              min_pruned_keyphrases, max_pruned_keyphrases,
                              min_link_significance,
                              infer_print_freq);
  }

  ifstream input(input_file);
  ofstream output(output_file);
  if (input.is_open()) {
    string new_line;
    bool file_load_complete = false;
    int chunk_size = batch_size / omp_get_max_threads() / 10;
    while (true) {
      vector<string> pool(batch_size);
      vector<string> to_output(batch_size);
      int size;
      for (size = 0; size < batch_size; ++size)
      {
        if (getline(input, new_line)) {
          pool[size] = new_line;
        } else {
          file_load_complete = true;
          break;
        }
      }

      #pragma omp parallel for schedule(dynamic, chunk_size)
      for (int i = 0; i < size; ++i) {
        string line = pool[i];
        SparseVector sparse_vector;
        encoders[omp_get_thread_num()]->encode(line, &sparse_vector);
        to_output[i] = sparse_vector.str();
      }

      for (int i = 0; i < size; ++i) {
        output << to_output[i] << endl;
      }
      if (file_load_complete) {
        break;
      }
    }
    input.close();
  }
  output.close();
}

