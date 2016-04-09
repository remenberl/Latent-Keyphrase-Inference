#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "parser/segphrase_parser.h"
#include "tools/easyloggingpp.h"
#include "tools/stringhelper.h"

using namespace std;

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
  if (strcmp(argv[argc - 1], "--help") == 0 || strcmp(argv[argc - 1], "-help") == 0) {
    cerr << "[usage] -input <input text snippets> "
         << "-segphrase <segphrase model> "
         << "-keyphrase <keyphrase file> "
         << "-num <number of keyphrases> "
         << "-output <output content units>" << endl;
    return -1;
  }

  string input_file_path, output_file_path, segphrase_model_path, keyphrase_file_path;
  int num_keyphrases = 50000;
  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-input") == 0) {
      input_file_path = argv[i + 1];
    } else if (strcmp(argv[i], "-segphrase") == 0) {
      segphrase_model_path = argv[i + 1];
    } else if (strcmp(argv[i], "-concept") == 0) {
      keyphrase_file_path = argv[i + 1];
    } else if (strcmp(argv[i], "-num") == 0) {
      num_keyphrases = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-output") == 0) {
      output_file_path = argv[i + 1];
    }
  }

  SegPhraseParser parser(segphrase_model_path, keyphrase_file_path, num_keyphrases);

  ifstream input_file(input_file_path);
  ofstream output_file(output_file_path);
  if (input_file.is_open()) {
    string line;
    while (getline(input_file, line)) {
      vector<string> elements = parser.segment(line);

      for (auto element: elements) {
        replace(element.begin(), element.end(), ' ', '_');
        output_file << element << " ";
      }

      output_file << endl;
    }
    input_file.close();
  }
  output_file.close();
}

