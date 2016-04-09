#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "parser/segphrase_parser.h"
#include "tools/easyloggingpp.h"
#include "tools/inireader.h"
#include "tools/stringhelper.h"

using namespace std;

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv)
{
  if (strcmp(argv[argc - 1], "--help") == 0 || strcmp(argv[argc - 1], "-help") == 0) {
    cerr << "[usage] -input <input text snippets> "
         << "-vocab <vocabulary path> "
         << "-prefix <keyphrase start id> "
         << "-segphrase <segphrase model> "
         << "-keyphrase <keyphrase file> "
         << "-num <number of keyphrases> "
         << "-output <output content units ids>" << endl;
    return -1;
  }

  string input_file_path, vocabulary_file_path, output_file_path, segphrase_model_path, keyphrase_file_path;
  int keyphrase_start_id = 100000000;
  int num_keyphrases = 50000;
  for (int i = 1; i < argc; i += 2) {
    if (strcmp(argv[i], "-input") == 0) {
      input_file_path = argv[i + 1];
    } else if (strcmp(argv[i], "-vocab") == 0) {
      vocabulary_file_path = argv[i + 1];
    } else if (strcmp(argv[i], "-prefix") == 0) {
      keyphrase_start_id = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-segphrase") == 0) {
      segphrase_model_path = argv[i + 1];
    } else if (strcmp(argv[i], "-keyphrase") == 0) {
      keyphrase_file_path = argv[i + 1];
    } else if (strcmp(argv[i], "-num") == 0) {
      num_keyphrases = atoi(argv[i + 1]);
    } else if (strcmp(argv[i], "-output") == 0) {
      output_file_path = argv[i + 1];
    }
  }

  SegPhraseParser parser(segphrase_model_path, keyphrase_file_path, num_keyphrases);

  ifstream vocabulary_file(vocabulary_file_path);
  unordered_map<string, int> vocabulary;
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      if (stoi(elements[0]) >= keyphrase_start_id) {
        vocabulary[elements[1]] = stoi(elements[0]);
      }
    }
    vocabulary_file.close();
  }

  ifstream input_file(input_file_path);
  ofstream output_file(output_file_path);
  if (input_file.is_open()) {
    string line;
    while (getline(input_file, line)) {
      vector<string> elements = parser.segment(line);

      set<int> units;
      for (auto element: elements) {
        if (vocabulary.find(element) != vocabulary.end()) {
          units.insert(vocabulary.at(element));
        }
      }

      if (elements.size() >= 2) {
        for (unsigned int j = 2; j < 4; ++j) {
          for (unsigned int i = 0; i < elements.size() - j + 1; ++i) {
            string combined_unit = elements[i];
            for (unsigned int k = 1; k < j; ++k) {
              combined_unit += " " + elements[i + k];
            }
            if (vocabulary.find(combined_unit) != vocabulary.end()) {
              units.insert(vocabulary.at(combined_unit));
            }
          }
        }
      }

      for (const auto& unit: units) {
        output_file << unit << " ";
      }
      output_file << endl;
    }
    input_file.close();
  }
  output_file.close();
}

