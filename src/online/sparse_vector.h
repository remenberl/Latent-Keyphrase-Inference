#ifndef __SPARSE_VECTOR_H__
#define __SPARSE_VECTOR_H__

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

#include "tools/stringhelper.h"

using namespace std;

struct IndexValuePair
{
  int index;
  double value;
  
  IndexValuePair(int index, double value) : index(index), value(value) {
  }
};

struct SparseVector
{
  vector< IndexValuePair > values;
  
  SparseVector() {}
  SparseVector(string line) {
    vector<string> elements = split(line, ' ');
    values.clear();
    if (elements.size() >= 2) {
      for (unsigned int i = 1; i < elements.size(); ++i) {
        if (elements[i].find(':') != string::npos) {
          push_back(stoi(elements[i].substr(0, elements[i].find(':'))), 
                    stod(elements[i].substr(elements[i].find(':') + 1)));
        }
      }
    }
  }

  void push_back(int index, double value) {
    values.push_back(IndexValuePair(index, value));
  }
  
  void sort() {
    auto byIndex = [](const IndexValuePair &a, const IndexValuePair &b) {
      return a.index < b.index;
    };
    std::sort(values.begin(), values.end(), byIndex);
  }

  void clear() {
    values.clear();
  }

  void normalize() {
    double sum = 0;
    for (const auto& i: values) {
      sum += i.value * i.value;
    }
    sum = sqrt(sum);
    for (auto& i: values) {
      i.value = i.value / sum;
    }
  }

  string str() const{
    stringstream output;
    output << values.size();
    for (const auto& element: values) {
      output << " " << element.index << ":" << element.value;
    }
    return output.str();
  }
  
  friend ostream &operator<<(ostream &output, const SparseVector &sparse_vector) {
    output << sparse_vector.str();
    return output;
  }
};

#endif
