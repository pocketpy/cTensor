import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

OPERATOR_FILE_PATH = os.path.join(PROJECT_ROOT, "src", "operator.c")
TEST_DIR_PATH = os.path.join(PROJECT_ROOT, "tests", "Operator")

def get_defined_operators(operator_file):
    """Extracts operator names (Tensor_XXX) from the operator source file."""
    operators = set()
    try:
        with open(operator_file, 'r') as f:
            content = f.read()
            matches = re.findall(r"(?:Tensor|static\s+Tensor)\s+(Tensor_([a-zA-Z0-9_]+))\s*\(", content)
            for match in matches:
                operators.add(match[1])
    except FileNotFoundError:
        print(f"Error: Operator file not found: {operator_file}", file=sys.stderr)
        sys.exit(1)
    return operators

def get_existing_test_files(test_dir):
    """Lists existing test files (test_xxx.c) in the specified directory."""
    test_files = set()
    try:
        for filename in os.listdir(test_dir):
            if filename.startswith("test_") and filename.endswith(".c"):
                test_name = filename[len("test_"):-len(".c")]
                test_files.add(test_name)
    except FileNotFoundError:
        print(f"Error: Test directory not found: {test_dir}", file=sys.stderr)
        sys.exit(1)
    return test_files

def main():
    print(f"Checking operator test coverage...")
    print(f"Operator source file: {os.path.abspath(OPERATOR_FILE_PATH)}")
    print(f"Test directory: {os.path.abspath(TEST_DIR_PATH)}")

    defined_operators = get_defined_operators(OPERATOR_FILE_PATH)
    if not defined_operators:
        print("No operators found in operator file. Exiting.", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(defined_operators)} operators: {sorted(list(defined_operators))}")

    existing_test_files = get_existing_test_files(TEST_DIR_PATH)
    print(f"Found {len(existing_test_files)} test files: {sorted(list(existing_test_files))}")

    missing_tests = []
    for op_name in defined_operators:
        if op_name not in existing_test_files:
            missing_tests.append(op_name)

    if not missing_tests:
        print("\nAll defined operators have corresponding test files.")
        sys.exit(0)
    else:
        print("\nError: The following operators are missing test files in", TEST_DIR_PATH + ":", file=sys.stderr)
        for missing_op in sorted(missing_tests):
            print(f"  - Operator: Tensor_{missing_op} (Expected test file: test_{missing_op}.c)", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
