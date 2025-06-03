#include "csv_reporter.h"
#include "test_config.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

static FILE* csv_file = NULL;

int csv_reporter_init(const char* filename) {
    if (csv_file != NULL) {
        fclose(csv_file);
    }
    csv_file = fopen(filename, "w");
    if (csv_file == NULL) {
        perror("Error opening CSV report file");
        return -1;
    }
    // Write header - Operator, TestPoint, ResultDetail
    fprintf(csv_file, "Operator,TestPoint,ResultDetail\n");
    fflush(csv_file);
    return 0;
}

void csv_reporter_add_entry(const char* operator_name, const char* test_case_identifier, bool passed, const char* failure_detail) {
    if (csv_file == NULL) {
        fprintf(stderr, "CSV reporter not initialized.\n");
        return;
    }

    fprintf(csv_file, "%s,%s,", operator_name, test_case_identifier);

    if (passed) {
        fprintf(csv_file, "/\n");
    } else {
        if (failure_detail != NULL && strlen(failure_detail) > 0) {
            fprintf(csv_file, "%s\n", failure_detail);
        } else {
            fprintf(csv_file, "failed_unknown_detail_on_%s\n", PLATFORM_NAME);
        }
    }
    fflush(csv_file);
}

void csv_reporter_close() {
    if (csv_file != NULL) {
        fclose(csv_file);
        csv_file = NULL;
    }
}
