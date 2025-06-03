#ifndef CSV_REPORTER_H
#define CSV_REPORTER_H

#include <stdio.h>
#include <stdbool.h>

// Returns 0 on success, -1 on failure.
int csv_reporter_init(const char* filename);

// Appends a test result to the CSV report.

// operator_name: Name of the cTensor operator (e.g., "add", "mul").

// test_case_identifier: A unique string identifying the specific test case or aspect being tested
//                       (e.g., "add_scalar", "add_vector_shape_check", "add_matrix_element_0_0").

// passed: true if the test aspect passed, false if failed.

// failure_detail: If 'passed' is false, this string contains the failure information,
//                 formatted as "observed/expected/platform" for value mismatches,
//                 or a descriptive string for other failures (e.g., "shape_mismatch").
//                 If 'passed' is true, this argument is ignored.

void csv_reporter_add_entry(const char* operator_name, const char* test_case_identifier, bool passed, const char* failure_detail);
void csv_reporter_close();

#endif
