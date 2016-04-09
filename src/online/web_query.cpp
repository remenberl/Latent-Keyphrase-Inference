#include <netinet/in.h>    // for sockaddr_in
#include <sys/types.h>    // for socket
#include <sys/socket.h>    // for socket
#include <stdio.h>        // for printf
#include <stdlib.h>        // for exit
#include <string.h>        // for bzero
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "model/model.h"
#include "model/node.h"
#include "online/encoder.h"
#include "tools/easyloggingpp.h"
#include "tools/stringhelper.h"

#define SERVER_PORT    2424
#define LENGTH_OF_LISTEN_QUEUE 20
#define BUFFER_SIZE 1024000

using namespace std;

_INITIALIZE_EASYLOGGINGPP

bool SortByEnergyDesc(const pair<int, double>& pair1, const pair<int, double>& pair2)
{
  return pair1.second > pair2.second;
}

int main(int argc, char **argv)
{
  _START_EASYLOGGINGPP(argc, argv);

  string vocabulary_path = "tmp/dblp/vocabulary.txt";
  string model_path = "tmp/dblp/model.now";

  unordered_map<int, string> vocabulary;
  ifstream vocabulary_file(vocabulary_path);
  if (vocabulary_file.is_open()) {
    string line;
    getline(vocabulary_file,line);

    while (getline(vocabulary_file, line)) {
      vector<string> elements = split(line, '\t');
      vocabulary[stoi(elements[0])] = elements[1];
    }
    vocabulary_file.close();
  }

  string segphrase_path = "tmp/dblp/segmentation.model";
  string keyphrase_path = "tmp/dblp/keyphrases.csv";
  Encoder encoder(model_path, segphrase_path, keyphrase_path, 40000,
                  vocabulary_path, 100000000, 60, 3000, 10, 200, 1e-3, 1);
  Model *model = encoder.getModel();
  SparseVector sparse_vector;

  struct sockaddr_in server_addr;
  bzero(&server_addr,sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_addr.s_addr = htons(INADDR_ANY);
  server_addr.sin_port = htons(SERVER_PORT);

  int server_socket = socket(PF_INET,SOCK_STREAM,0);
  if( server_socket < 0)
  {
      LOG(ERROR) << "Create Socket Failed!";
      exit(1);
  }

  int opt =1;
  setsockopt(server_socket,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
  if( bind(server_socket,(struct sockaddr*)&server_addr,sizeof(server_addr)))
  {
      LOG(ERROR) << "Server Bind Port : %d Failed!" << SERVER_PORT;
      exit(1);
  }

  if ( listen(server_socket, LENGTH_OF_LISTEN_QUEUE) )
  {
      LOG(ERROR) << "Server Listen Failed!";
      exit(1);
  }
  while (1)
  {
    struct sockaddr_in client_addr;
    socklen_t length = sizeof(client_addr);
    int new_server_socket = accept(server_socket,(struct sockaddr*)&client_addr,&length);
    if (new_server_socket < 0)
    {
        LOG(ERROR) << "Server Accept Failed!";
        break;
    }
    char buffer[BUFFER_SIZE];
    bzero(buffer, BUFFER_SIZE);
    length = recv(new_server_socket,buffer,BUFFER_SIZE,0);
    if (length < 0)
    {
        LOG(ERROR) << "Server Recieve Data Failed!\n";
        break;
    }
    ostringstream ss;

    if (string(buffer).substr(0, 9) == "__mode1__") {
      vector<string> elements = encoder.segment(string(buffer).substr(9));
      if (string(buffer).size() < 10) {
        ss << "Sorry, query is invalid.";
      } else {
        for (unsigned int i = 0; i < elements.size() - 1; ++i) {
          ss << elements[i] << " / ";
        }
        if (elements.size() > 0) {
          ss << elements[elements.size() - 1];
          ss << "<br><br><i>Please move curser onto specific keyphrase in the left table to view its silhouette.</i>";
        } else {
          ss << "Sorry, query is invalid.";
        }
      }
    } else {
      LOG(INFO) << "Query: " << string(buffer).substr(9);
      encoder.encode(string(buffer).substr(9), &sparse_vector);
      vector<pair<int, double>> node_energy;
      for (auto node: sparse_vector.values) {
        node_energy.push_back(make_pair(node.index, node.value));
      }
      sort(node_energy.begin(), node_energy.end(), SortByEnergyDesc);

      if (node_energy.size() == 0) {
        ss << "Sorry, we currently only support CS domain queries in database, "
           << "data mining, computer vision, machine learning and natural language processing.";
      } else {
        for (auto node: node_energy) {
          vector<string> print_vector;
          model->GetNode(node.first)->PrintChildren(vocabulary, &print_vector, 20);
          if (vocabulary.find(node.first) != vocabulary.end()) {
            ss << "<tr><td data-original-title=\"Keyphrase Silhouette<br>";
            for (unsigned int i = 0; i < print_vector.size() - 1; ++i) {
              ss << print_vector[i] << ", ";
            }
            if (print_vector.size() > 0) {
              ss << print_vector[print_vector.size() - 1];
            }
            ss << " ...";
            ss << "\" data-container=\"body\""
               << "data-toggle=\"tooltip\" data-placement=\"bottom\" data-html=\"true\" title=\"\">"
               << vocabulary[node.first] << "</td><td>"
               << node.first << "</td><td>" << node.second << "</td></tr>";
          }
        }
      }
    }
    if(send(new_server_socket,ss.str().c_str(),ss.str().length() + 1,0)<0)
    {
      close(new_server_socket);
      break;
    }

    close(new_server_socket);
  }

  close(server_socket);
  return 0;
}
