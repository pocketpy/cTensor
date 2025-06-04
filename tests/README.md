# Running cTensor Tests

This guide provides instructions on how to build and run the test suite for the cTensor library. Our tests are designed to ensure the correctness and reliability of the tensor operations and neural network functionalities.

## Prerequisites

Before you begin, make sure you have the following installed on your system:

*   **CMake**: Version 3.10 or higher. CMake is used to configure and generate the build system.
*   **C Compiler**: A C11-compliant compiler such as GCC, Clang, or MSVC (for Windows).

## Building the Tests

The tests are built as part of the main cTensor project using CMake. Follow these steps:

1.  **Navigate to the Root Directory**: Open your terminal and change to the root directory of the cTensor project.
    ```bash
    cd /path/to/cTensor
    ```

2.  **Configure CMake**: Create a build directory (e.g., `build`) and run CMake to configure the project. This step prepares the build files.
    ```bash
    cmake -B build -S .
    ```
    *   The `-B build` option specifies that the build files should be generated in a subdirectory named `build`.
    *   The `-S .` option indicates that the source directory is the current directory (the project root).

3.  **Build the Test Executable**: Compile the project, specifically targeting the test executable `cten_tests`.
    ```bash
    cmake --build build --target cten_tests
    ```
    This command will compile the necessary source files and link them to create the `cten_tests` executable. On Windows, this will be `cten_tests.exe`.

## Running the Tests

Once the tests are built, you can run them in a couple of ways:

### Option 1: Using CTest (Recommended)

CTest is CMake's testing tool and is the recommended way to run tests as it integrates well with the CMake build process.

1.  **Navigate to the Build Directory**:
    ```bash
    cd build
    ```

2.  **Run CTest**:
    ```bash
    ctest -C Debug --output-on-failure
    ```
    *   `-C Debug` specifies the build configuration (can be `Release` or other configurations if set up). Adjust if you used a different configuration.
    *   `--output-on-failure` will display the output from tests only if they fail, keeping the console clean for successful runs.

### Option 2: Running the Executable Directly

You can also run the compiled test executable directly.

1.  The executable is typically located in a subdirectory within your `build` folder. The exact path might vary slightly based on your system and CMake generator:
    *   On Linux/macOS: `build/bin/cten_tests` or `build/cten_tests`
    *   On Windows: `build\bin\Debug\cten_tests.exe` or `build\Debug\cten_tests.exe`

2.  **Execute the test program**:
    For example, on Linux/macOS:
    ```bash
    ./build/bin/cten_tests
    ```
    Or on Windows (from the project root):
    ```powershell
    .\build\bin\Debug\cten_tests.exe
    ```

## Test Reports

After running the tests, a CSV file named `cten_test_report.csv` is generated in the `build` directory. This report contains a summary of the test results in a multi-column format that includes:

- **Operator**: The name of the operator being tested (e.g., "add")
- **TestPoint**: The specific test case name (e.g., "add_scalar")
- **1, 2, 3, ...**: Numbered columns for sub-tests within each test case

Passing tests are marked with a `/` symbol, while failing tests include detailed error information.

## CSV Reporting System

The cTensor test suite uses a buffered CSV reporter system that stores test results in memory until all tests complete. The API includes:

```c
int csv_reporter_init(const char *filename);
void csv_reporter_record_result(const char *operator_name, const char *test_point_name, 
                              int sub_test_index, const char *result_detail);
void csv_reporter_close();
```

## Adding New Test Cases

Follow these steps to add a new operator test:

1. Create a new file `test_<operator>.c` in the `tests/Operator/` directory
2. Add a function declaration `void test_<operator>_operator();` in `tests/cten_tests.c`
3. Call this function from `main()` in `tests/cten_tests.c`

### Structure of Test Files

Each test file should follow this pattern:

```c
#include "../test_utils.h"

void test_<operator>_operator() {
    const char* op_name = "<operator>";
    PoolId pool_id = 0;
    cten_begin_malloc(pool_id);
    
    // Test Case 1
    {
        const char* tc_name = "<operator>_<test_case>";
        /* Create test tensors, perform operations */
        
        // Use explicit sub-test indexing
        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 1, TEST_FLOAT_TOLERANCE);
    }
    
    // Multiple sub-tests for the same test case use the same tc_name but different sub_test_index values
    {
        const char* tc_name = "<operator>_<test_case>";
        /* Different test parameters */
        
        compare_tensors(&actual_res, &expected_res, op_name, tc_name, 2, TEST_FLOAT_TOLERANCE);
    }
    
    cten_end_malloc(pool_id);
}
```

### Important Notes

- Always specify explicit sub-test indices (1, 2, 3, etc.) when calling `compare_tensors()`
- Use the same `tc_name` for different sub-tests of the same test case
- The `compare_tensors()` function handles CSV reporting internally; don't make separate calls to `csv_reporter_record_result()`


---

Happy testing! If you encounter any issues or have suggestions, please feel free to open an issue in the project repository.
