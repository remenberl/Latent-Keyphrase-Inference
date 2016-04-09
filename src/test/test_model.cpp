#include <string>

#include "model/model.h"
#include "tools/easyloggingpp.h"

_INITIALIZE_EASYLOGGINGPP

int main(int argc, char** argv) 
{
  string model_path = "tmp/dblp/model.init";

  Model model(model_path, 100000000);
  model.Dump("./tmp/dblp/model.test");
}