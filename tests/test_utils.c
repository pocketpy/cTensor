#include "test_utils.h"
#include "test_config.h"
#include "csv_reporter.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

bool compare_floats(float a, float b, float tolerance) { return fabs(a - b) < tolerance; }

Tensor create_test_tensor(TensorShape shape, float* data, bool requires_grad) {
    Tensor t = Tensor_new(shape, requires_grad);
    if(t.data != NULL && data != NULL) {
        memcpy(t.data->flex, data, t.data->numel * sizeof(float));
    }
    return t;
}

void print_tensor(const Tensor* t, const char* name) {
    if(name) {
        printf("Tensor %s (Shape: (", name);
    } else {
        printf("Tensor (Shape: (");
    }
    TensorShape shape_copy_for_print;
    memcpy(shape_copy_for_print, t->shape, sizeof(TensorShape));
    for(int i = 0; i < CTENSOR_MAX_DIMS && shape_copy_for_print[i] != 0; ++i) {
        printf("%d%s",
               shape_copy_for_print[i],
               (shape_copy_for_print[i + 1] != 0 && i < CTENSOR_MAX_DIMS - 1) ? ", " : "");
    }
    printf(")):\n");

    if(t->data == NULL) {
        printf("  [Data is NULL]\n");
        return;
    }
    if(t->data->numel == 0) {
        printf("  [Empty tensor]\n");
        return;
    }

    for(size_t i = 0; i < t->data->numel; ++i) {
        printf("%.4f ", t->data->flex[i]);
        if(t->shape[1] != 0 && (i + 1) % t->shape[1] == 0) { printf("\n"); }
    }
    if(t->shape[1] == 0 && t->data->numel > 0) { printf("\n"); }
    printf("\n");
}

bool compare_tensors(const Tensor* t_observed,
                     const Tensor* t_expected,
                     const char* operator_name,
                     const char* test_point_name,
                     int sub_test_index,
                     float tolerance) {
    char failure_detail_buffer[512];

    if(t_observed == NULL || t_expected == NULL) {
        const char* detail = "observed_is_NULL";
        if(t_observed == NULL && t_expected == NULL)
            detail = "both_are_NULL";
        else if(t_expected == NULL)
            detail = "expected_is_NULL";
        snprintf(failure_detail_buffer,
                 sizeof(failure_detail_buffer),
                 "%s/%s/%s",
                 detail,
                 "not_null",
                 PLATFORM_NAME);
        csv_reporter_record_result(operator_name,
                                   test_point_name,
                                   sub_test_index,
                                   failure_detail_buffer);
        return false;
    }

    // 1. Compare dimensions
    TensorShape shape_obs_copy_for_dim, shape_exp_copy_for_dim;
    memcpy(shape_obs_copy_for_dim, t_observed->shape, sizeof(TensorShape));
    memcpy(shape_exp_copy_for_dim, t_expected->shape, sizeof(TensorShape));
    int dim_obs = TensorShape_dim(shape_obs_copy_for_dim);
    int dim_exp = TensorShape_dim(shape_exp_copy_for_dim);
    if(dim_obs != dim_exp) {
        snprintf(failure_detail_buffer,
                 sizeof(failure_detail_buffer),
                 "%d/%d/%s_dim_mismatch",
                 dim_obs,
                 dim_exp,
                 PLATFORM_NAME);
        csv_reporter_record_result(operator_name,
                                   test_point_name,
                                   sub_test_index,
                                   failure_detail_buffer);
        return false;
    }

    // 2. Compare shapes
    for(int i = 0; i < dim_obs; ++i) {
        if(t_observed->shape[i] != t_expected->shape[i]) {
            snprintf(failure_detail_buffer,
                     sizeof(failure_detail_buffer),
                     "%d/%d/%s_shape_mismatch_at_dim%d",
                     t_observed->shape[i],
                     t_expected->shape[i],
                     PLATFORM_NAME,
                     i);
            csv_reporter_record_result(operator_name,
                                       test_point_name,
                                       sub_test_index,
                                       failure_detail_buffer);
            return false;
        }
    }

    // Check for NULL data buffers
    if(t_observed->data == NULL || t_expected->data == NULL) {
        const char* detail = "observed_data_is_NULL";
        if(t_observed->data == NULL && t_expected->data == NULL && t_observed->shape[0] == 0)
            ;
        else if(t_observed->data == NULL && t_expected->data == NULL)
            detail = "both_data_are_NULL";
        else if(t_expected->data == NULL)
            detail = "expected_data_is_NULL";
        else {
            snprintf(failure_detail_buffer,
                     sizeof(failure_detail_buffer),
                     "%s/%s/%s",
                     detail,
                     "non-NULL_data_expected_or_vice_versa",
                     PLATFORM_NAME);
            csv_reporter_record_result(operator_name,
                                       test_point_name,
                                       sub_test_index,
                                       failure_detail_buffer);
            return false;
        }
    }

    // Handle case where one data is NULL but other is not (and not an empty tensor case)
    if((t_observed->data == NULL && t_expected->data != NULL && t_expected->data->numel > 0) ||
       (t_observed->data != NULL && t_expected->data == NULL && t_observed->data->numel > 0)) {
        const char* detail = (t_observed->data == NULL) ? "observed_data_NULL_expected_not_NULL"
                                                        : "observed_data_not_NULL_expected_NULL";
        snprintf(failure_detail_buffer,
                 sizeof(failure_detail_buffer),
                 "%s/%s/%s",
                 detail,
                 "data_buffer_discrepancy",
                 PLATFORM_NAME);
        csv_reporter_record_result(operator_name,
                                   test_point_name,
                                   sub_test_index,
                                   failure_detail_buffer);
        return false;
    }

    // If both data buffers are NULL (e.g. for 0-element tensors), numel should be 0 for both.
    size_t numel_obs = (t_observed->data) ? t_observed->data->numel : 0;
    size_t numel_exp = (t_expected->data) ? t_expected->data->numel : 0;

    // 3. Compare number of elements
    if(numel_obs != numel_exp) {
        snprintf(failure_detail_buffer,
                 sizeof(failure_detail_buffer),
                 "%zu/%zu/%s_numel_mismatch",
                 numel_obs,
                 numel_exp,
                 PLATFORM_NAME);
        csv_reporter_record_result(operator_name,
                                   test_point_name,
                                   sub_test_index,
                                   failure_detail_buffer);
        return false;
    }

    // If numel is 0, and all previous checks passed, tensors are considered equal (e.g. two empty
    // tensors of same shape)
    if(numel_obs == 0) { return true; }

    // 4. Compare data element-wise (only if data buffers are not NULL and numel > 0)
    for(size_t i = 0; i < numel_obs; ++i) {
        if(!compare_floats(t_observed->data->flex[i], t_expected->data->flex[i], tolerance)) {
            snprintf(failure_detail_buffer,
                     sizeof(failure_detail_buffer),
                     "%.*g/%.*g/%s",
                     15,
                     t_observed->data->flex[i],
                     15,
                     t_expected->data->flex[i],
                     PLATFORM_NAME);
            csv_reporter_record_result(operator_name,
                                       test_point_name,
                                       sub_test_index,
                                       failure_detail_buffer);
            return false;  // Fail on first mismatch
        }
    }

    // All tests passed, record success
    csv_reporter_record_result(operator_name, test_point_name, sub_test_index, "/");
    return true;
}
