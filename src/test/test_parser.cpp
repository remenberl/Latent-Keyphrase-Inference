#include "parser/segphrase_parser.h"

template<class T>
void printVector(vector<T> a) {
    for (size_t i = 0; i < a.size(); ++ i) {
        cerr << a[i];
        if (i + 1 == a.size()) {
            cerr << endl;
        } else {
            cerr << ", ";
        }
    }
}

int main()
{
    string model_path = "tmp/dblp/segmentation.model";
    string keyphrase_path = "tmp/dblp/keyphrases.csv";
    SegPhraseParser* parser = new SegPhraseParser(model_path, keyphrase_path, 50000);
    cerr << "parser built." << endl;

    vector<string> segments = parser->segment("data mining is an area");
    printVector(segments);

    cerr << "Please type in a sentence in a single line (or exit()):" << endl;
    while (getLine(stdin)) {
        if (strcmp(line, "exit()") == 0) {
            break;
        }
        segments = parser->segment(line);
        cerr << "[Segmentation Result]" << endl;
        printVector(segments);
        cerr << "\nPlease type in a sentence in a single line (or exit()):" << endl;
    }

    cerr << "[done]" << endl;
    return 0;
}