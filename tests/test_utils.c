#include "test_utils.h"
#include <string.h>
#include <math.h> 
#include <stdio.h> 

// Function to create a Tensor from a flat array of floats and a given shape
Tensor create_tensor(const float* data, TensorShape shape, bool requires_grad) {
    Tensor t = Tensor_new(shape, requires_grad);
    int num_elements = TensorShape_numel(shape);
    if (t.data != NULL && data != NULL) {
        memcpy(t.data->flex, data, num_elements * sizeof(float));
    }
    return t;
}

// Function to compare two tensors for equality (within a small tolerance for floats)
bool compare_tensors(Tensor t1, Tensor t2, float tolerance) {
    // Compare shapes (dimensionality and size of each dimension)
    for (int i = 0; i < 4; ++i) {
        if (t1.shape[i] != t2.shape[i]) {
            int t1_dim = TensorShape_dim(t1.shape);
            int t2_dim = TensorShape_dim(t2.shape);
            if (i < t1_dim || i < t2_dim) { // Only fail if it's a meaningful dimension for either
                 if (t1.shape[i] != t2.shape[i]) return false;
            }
        }
    }

    // Compare number of elements
    if (t1.data->numel != t2.data->numel) {
        return false;
    }

    // Compare data elements
    for (int i = 0; i < t1.data->numel; ++i) {
        if (fabs(t1.data->flex[i] - t2.data->flex[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Function to print a tensor's shape and data
void print_tensor_data(Tensor t) {
    char shape_str[100];
    TensorShape_tostring(t.shape, shape_str, sizeof(shape_str));
    printf("Tensor Shape: %s\n", shape_str);
    printf("Tensor Data (%d elements):\n[", t.data->numel);
    for (int i = 0; i < t.data->numel; ++i) {
        printf("%.4f", t.data->flex[i]);
        if (i < t.data->numel - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}
