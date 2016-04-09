// #include "tools/math.h"
#include <cmath>
#include <vector>

using namespace std;

//log(1-exp(-a))
double log1mexp(double a) {
	if (a > 0.693) {
		return log1p(-exp(-a));
	} else {
		return log(-expm1(-a));
	}
	// naive: return log(1-exp(-a))
}

//log(sum of exp(a))
double logsumexp(double nums[], double sign[], int ct) {
  double max_exp = nums[0], sum = 0.0;
  int i;

  for (i = 1 ; i < ct ; i++)
    if (nums[i] > max_exp)
      max_exp = nums[i];

  for (i = 0; i < ct ; i++)
    sum += sign[i] * exp(nums[i] - max_exp);

  return log(sum) + max_exp;

  // naive:
  // double sum = 0.0;
  // for (i = 0; i < ct ; i++)
  //   sum += sign[i] * exp(nums[i]);

  // return log(sum);
}


double logsumexp(vector<double> nums) {
  double max_exp = nums[0], sum = 0.0;

  for (const auto& num: nums)
    if (num > max_exp)
      max_exp = num;

  for (const auto& num: nums)
    sum += exp(num - max_exp);

  return log(sum) + max_exp;  
}