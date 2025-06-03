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

After running the tests, a CSV file named `cten_test_report_<platform>.csv` (e.g., `cten_test_report_linux.csv`) is generated in the `build` directory. This report contains a summary of the test results, including which tests passed or failed.

This report is used by our Continuous Integration (CI) system to check the health of the codebase.

---

Happy testing! If you encounter any issues or have suggestions, please feel free to open an issue in the project repository.
