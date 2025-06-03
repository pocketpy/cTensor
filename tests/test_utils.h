#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "../include/cten.h"
#include <stdbool.h>

bool compare_floats(float a, float b, float tolerance);
bool compare_tensors(const Tensor* t_observed, const Tensor* t_expected, const char* operator_name, const char* base_test_case_name, float tolerance);
Tensor create_test_tensor(TensorShape shape, float* data, bool requires_grad);
void print_tensor(const Tensor* t, const char* name);

#endif
