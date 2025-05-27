#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include "cten.h"
#include <stdbool.h>
#include <stdio.h>  

Tensor create_tensor(const float* data, TensorShape shape, bool requires_grad);
bool compare_tensors(Tensor t1, Tensor t2, float tolerance);
void print_tensor_data(Tensor t);

#endif 
