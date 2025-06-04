#ifndef CSV_REPORTER_H
#define CSV_REPORTER_H

#include <stdio.h> 

// Returns 0 on success, -1 on failure.
int csv_reporter_init(const char *filename);

// Records a test result for a specific sub-test within a test point.
// operator_name: Name of the operator being tested (e.g., "add").

// test_point_name: The base name for the test case, defining a row in the CSV (e.g., "add_scalar").

// sub_test_index: A 1-based index specifying the sub-test column for this result (e.g., 1, 2, 3...).
// result_detail: A string describing the result (e.g., "/" for pass, or "observed/expected/platform" for fail).
void csv_reporter_record_result(const char *operator_name, const char *test_point_name, int sub_test_index, const char *result_detail);
void csv_reporter_close();

#endif
