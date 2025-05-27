import os
import sys
import glob

EXPECTED_PASSES = [
    "Test on Tensor_add Operator: PASS",
    "Test on Tensor_matmul Operator: PASS"
]

def check_file(filepath):
    platform = os.path.basename(os.path.dirname(filepath))
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"[FAIL] {platform}: Error reading file - {e}")
        return False
    
    missing_tests = []
    for expected_pass in EXPECTED_PASSES:
        if expected_pass not in content:
            missing_tests.append(expected_pass)
    
    failed_lines = []
    if "FAIL" in content:
        for line_num, line in enumerate(content.splitlines()):
            if "FAIL" in line:
                failed_lines.append(f"L{line_num+1}: {line.strip()}")
    
    if missing_tests or failed_lines:
        print(f"[FAIL] {platform}")
        for missing in missing_tests:
            print(f"       Missing: {missing}")
        for failed in failed_lines:
            print(f"       {failed}")
        return False
    else:
        print(f"[PASS] {platform}")
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_all_results.py <results_directory_path>")
        sys.exit(2)
    
    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: Directory '{results_dir}' not found")
        sys.exit(2)
    
    result_files = glob.glob(os.path.join(results_dir, "*", "results-*.txt"))
    if not result_files:
        print(f"No result files found in '{results_dir}'")
        sys.exit(1)
    
    print("Testing Results:")
    print("=" * 40)
    
    overall_success = True
    for result_file in sorted(result_files):
        if not check_file(result_file):
            overall_success = False
    
    print("=" * 40)
    if overall_success:
        print("Build Status: SUCCESS")
        sys.exit(0)
    else:
        print("Build Status: FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()