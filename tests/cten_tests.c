#include <stdio.h>
#include "../include/cten.h"
#include "csv_reporter.h"
#include "test_config.h"
#include <string.h>
#include <stdlib.h>

#if defined(_WIN32) || defined(_WIN64)
    #define PATH_SEPARATOR_CHAR '\\'
#else
    #define PATH_SEPARATOR_CHAR '/'
#endif

#define XSTR(s) STR(s)
#define STR(s) #s

void test_add_operator();
void test_mul_operator();
void test_sub_operator();
void test_mean_operator();
void test_matmul_operator();
void test_mulf_operator();
void test_sum_operator();
void test_pow_operator();
void test_reciprocal_operator();
void test_square_operator();
void test_div_operator();

// Backward tests
void test_add_backward();
void test_mul_backward();
void test_matmul_backward();
void test_sub_backward();
void test_relu_backward();
void test_linear_backward();
void test_sum_backward();
void test_mean_backward();

int main() {
    printf("Starting cTensor Test Suite on %s...\n", PLATFORM_NAME);

    cten_initilize();

    char report_path[512];
#ifdef CTEN_BUILD_DIR_PATH
    const char* build_dir = XSTR(CTEN_BUILD_DIR_PATH);
    char clean_build_dir[256];
    size_t len = strlen(build_dir);
    if (len > 1 && build_dir[0] == '"' && build_dir[len-1] == '"') {
        strncpy(clean_build_dir, build_dir + 1, len - 2);
        clean_build_dir[len - 2] = '\0';
    } else {
        strcpy(clean_build_dir, build_dir);
    }
    snprintf(report_path, sizeof(report_path), "%s%ccten_test_report_%s.csv", clean_build_dir, PATH_SEPARATOR_CHAR, PLATFORM_NAME);
#else
    snprintf(report_path, sizeof(report_path), "cten_test_report_%s.csv", PLATFORM_NAME);
#endif

    printf("Test report will be generated at: %s\n", report_path);

    if (csv_reporter_init(report_path) != 0) {
        fprintf(stderr, "Failed to initialize CSV reporter. Aborting tests.\n");
        cten_finalize();
        return 1;
    }

    test_add_operator();
    printf("Add operator tests finished.\n");

    test_mul_operator();
    printf("Mul operator tests finished.\n");

    test_sub_operator();
    printf("Sub operator tests finished.\n");

    test_mean_operator();
    printf("Mean operator tests finished.\n");

    test_matmul_operator();
    printf("Matmul operator tests finished.\n");

    test_mulf_operator();
    printf("Mulf operator tests finished.\n");

    test_sum_operator();
    printf("Sum operator tests finished.\n");

    test_pow_operator();
    printf("Pow operator tests finished.\n");

    test_reciprocal_operator();
    printf("Reciprocal operator tests finished.\n");

    test_square_operator();
    printf("Square operator tests finished.\n");

    test_div_operator();
    printf("Div operator tests finished.\n");

    // Backward tests
    test_add_backward();
    printf("Add backward tests finished.\n");

    test_mul_backward();
    printf("Mul backward tests finished.\n");

    test_matmul_backward();
    printf("Matmul backward tests finished.\n");
    
    test_sub_backward();
    printf("Sub backward tests finished.\n");
    
    test_relu_backward();
    printf("ReLU backward tests finished.\n");
    
    test_linear_backward();
    printf("Linear backward tests finished.\n");

    test_sum_backward();
    printf("Sum backward tests finished.\n");
        
    test_mean_backward();
    printf("Mean backward tests finished.\n");
    
    // other tests
    
    csv_reporter_close();
    cten_finalize();

    printf("cTensor Test Suite finished. Report generated: cten_test_report_%s.csv\n", PLATFORM_NAME);
    return 0;
}
