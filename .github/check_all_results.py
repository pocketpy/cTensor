import os
import sys
import glob

EXPECTED_PASSES = [
    "Test test_tensor_add: PASS",
    "Test test_tensor_matmul: PASS"
]

def check_file(filepath):
    print(f"Checking file: {filepath}...")
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

    all_tests_passed_in_file = True
    for expected_pass in EXPECTED_PASSES:
        if expected_pass not in content:
            print(f"  Missing expected result: '{expected_pass}'")
            all_tests_passed_in_file = False
    
    if "FAIL" in content:
        print(f"  File contains 'FAIL' indicating a test failure.")
        for line_num, line in enumerate(content.splitlines()):
            if "FAIL" in line:
                print(f"    L{line_num+1}: {line.strip()}")
        all_tests_passed_in_file = False

    if all_tests_passed_in_file:
        print(f"  All expected tests PASSED in {os.path.basename(filepath)}.")
    else:
        print(f"  One or more tests FAILED or were inconclusive in {os.path.basename(filepath)}.")
    
    return all_tests_passed_in_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_all_results.py <results_directory_path>")
        sys.exit(2) # Different exit code for usage error

    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: Provided path '{results_dir}' is not a directory.")
        sys.exit(2)

    print(f"Scanning for result files in: {results_dir}")
    result_files = glob.glob(os.path.join(results_dir, "*", "results-*.txt"))

    if not result_files:
        print(f"No result files found in subdirectories of '{results_dir}'. Pattern: */results-*.txt")
        sys.exit(1) # Fail if no results are found

    overall_success = True
    for result_file in result_files:
        if not check_file(result_file):
            overall_success = False

    if overall_success:
        print("\nAll tests passed successfully across all platforms and configurations!")
        sys.exit(0)
    else:
        print("\nOne or more tests failed. Please check the logs above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
