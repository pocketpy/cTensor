#include "csv_reporter.h"
#include "test_config.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define INITIAL_CAPACITY 16
#define MAX_SUB_TESTS 64

typedef struct {
    char* operator_name;
    char* test_point_name;
    char* results[MAX_SUB_TESTS];
} TestRow;

static TestRow* unique_rows_buffer = NULL;
static size_t unique_rows_count = 0;
static size_t unique_rows_capacity = 0;

static char csv_filename[FILENAME_MAX];
static int overall_max_sub_test_index = 0;

static char* cten_strdup(const char* s) {
    if(!s) return NULL;
    size_t len = strlen(s) + 1;
    char* new_s = (char*)malloc(len);
    if(new_s) { memcpy(new_s, s, len); }
    return new_s;
}

int csv_reporter_init(const char* filename) {
    if(unique_rows_capacity > 0 && unique_rows_buffer) {
        for(size_t i = 0; i < unique_rows_count; ++i) {
            free(unique_rows_buffer[i].operator_name);
            free(unique_rows_buffer[i].test_point_name);
            for(int j = 0; j < MAX_SUB_TESTS; ++j) {
                free(unique_rows_buffer[i].results[j]);
            }
        }
        free(unique_rows_buffer);
        unique_rows_buffer = NULL;
    }

    strncpy(csv_filename, filename, FILENAME_MAX - 1);
    csv_filename[FILENAME_MAX - 1] = '\0';

    unique_rows_buffer = (TestRow*)malloc(INITIAL_CAPACITY * sizeof(TestRow));
    unique_rows_capacity = unique_rows_buffer ? INITIAL_CAPACITY : 0;
    unique_rows_count = 0;
    overall_max_sub_test_index = 0;

    return unique_rows_buffer ? 0 : -1;
}

void csv_reporter_record_result(const char* operator_name,
                                const char* test_point_name,
                                int sub_test_index,
                                const char* result_detail) {
    if(!unique_rows_buffer || unique_rows_capacity == 0) {
        fprintf(
            stderr,
            "CSV Reporter: Record called before successful init or after failed init. Discarding result.\n");
        return;
    }

    if(sub_test_index <= 0 || sub_test_index > MAX_SUB_TESTS) {
        fprintf(stderr,
                "CSV Reporter: Invalid sub_test_index %d for %s - %s. Must be 1-%d. Discarding.\n",
                sub_test_index,
                operator_name,
                test_point_name,
                MAX_SUB_TESTS);
        return;
    }

    TestRow* row_to_update = NULL;

    // Find existing row
    for(size_t i = 0; i < unique_rows_count; ++i) {
        if(strcmp(unique_rows_buffer[i].operator_name, operator_name) == 0 &&
           strcmp(unique_rows_buffer[i].test_point_name, test_point_name) == 0) {
            row_to_update = &unique_rows_buffer[i];
            break;
        }
    }

    // If no existing row, create a new one
    if(!row_to_update) {
        if(unique_rows_count >= unique_rows_capacity) {
            size_t new_cap = unique_rows_capacity > 0 ? unique_rows_capacity * 2 : INITIAL_CAPACITY;
            TestRow* new_urb = (TestRow*)realloc(unique_rows_buffer, new_cap * sizeof(TestRow));
            if(!new_urb) {
                fprintf(
                    stderr,
                    "CSV Reporter: Failed to reallocate buffer for new row. Discarding result.\n");
                return;  // Cannot allocate more
            }
            unique_rows_buffer = new_urb;
            unique_rows_capacity = new_cap;
        }
        row_to_update = &unique_rows_buffer[unique_rows_count];
        row_to_update->operator_name = cten_strdup(operator_name);
        row_to_update->test_point_name = cten_strdup(test_point_name);
        if(!row_to_update->operator_name || !row_to_update->test_point_name) {
            fprintf(
                stderr,
                "CSV Reporter: Failed to allocate memory for new row names. Discarding result.\n");
            free(row_to_update->operator_name);
            free(row_to_update->test_point_name);
            return;
        }
        for(int k = 0; k < MAX_SUB_TESTS; ++k)
            row_to_update->results[k] = NULL;
        unique_rows_count++;
    }

    // Store result detail in the correct sub-test slot
    free(row_to_update->results[sub_test_index - 1]);
    row_to_update->results[sub_test_index - 1] = cten_strdup(result_detail);
    if(!row_to_update->results[sub_test_index - 1] && result_detail != NULL) {
        fprintf(
            stderr,
            "CSV Reporter: Failed to allocate memory for result detail. Result for %s-%s sub-test %d will be missing.\n",
            operator_name,
            test_point_name,
            sub_test_index);
    }

    if(sub_test_index > overall_max_sub_test_index) { overall_max_sub_test_index = sub_test_index; }
}

void csv_reporter_close() {
    if(!unique_rows_buffer) {
        fprintf(stderr, "CSV Reporter: Close called but not initialized or buffer is NULL.\n");
        return;
    }

    FILE* file = fopen(csv_filename, "w");
    if(!file) {
        perror("CSV Reporter: Error opening CSV file for writing");
        goto cleanup;  // Still try to free memory
    }

    fprintf(file, "Operator,TestPoint");
    for(int i = 1; i <= overall_max_sub_test_index; ++i) {
        fprintf(file, ",%d", i);
    }
    fprintf(file, "\n");

    for(size_t i = 0; i < unique_rows_count; ++i) {
        fprintf(file,
                "%s,%s",
                unique_rows_buffer[i].operator_name ? unique_rows_buffer[i].operator_name
                                                    : "ERROR_NULL_OP",
                unique_rows_buffer[i].test_point_name ? unique_rows_buffer[i].test_point_name
                                                      : "ERROR_NULL_TP");
        for(int j = 0; j < overall_max_sub_test_index;
            ++j) {  // Iterate up to overall_max_sub_test_index columns
            fprintf(file,
                    ",%s",
                    unique_rows_buffer[i].results[j] ? unique_rows_buffer[i].results[j] : "");
        }
        fprintf(file, "\n");
    }

    fclose(file);

cleanup:
    if(unique_rows_buffer) {
        for(size_t i = 0; i < unique_rows_count; ++i) {
            free(unique_rows_buffer[i].operator_name);
            free(unique_rows_buffer[i].test_point_name);
            for(int j = 0; j < MAX_SUB_TESTS; ++j) {
                free(unique_rows_buffer[i].results[j]);
            }
        }
        free(unique_rows_buffer);
        unique_rows_buffer = NULL;
    }
    unique_rows_count = 0;
    unique_rows_capacity = 0;
    overall_max_sub_test_index = 0;
}
