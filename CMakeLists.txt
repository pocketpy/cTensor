cmake_minimum_required(VERSION 3.10)
project(cTensor LANGUAGES C)

# Set C standard globally
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Include project headers
include_directories(include)

# Collect library sources (excluding main files)
file(GLOB_RECURSE LIB_SOURCES
    "src/*.c"
)
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*src2/main\.c$")

# Collect all sources for main executable
file(GLOB_RECURSE ALL_SOURCES "src/*.c" "src2/*.c")

# Create main executable
add_executable(cten_exe ${ALL_SOURCES})

# Add MSVC-specific compiler options for main executable
if(MSVC)
    target_compile_options(cten_exe PRIVATE /wd4305)
    target_compile_definitions(cten_exe PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Link math library (cross-platform)
if(NOT WIN32)
    target_link_libraries(cten_exe PRIVATE m)
endif()

# Testing setup
# Test utilities and main runner
set(TEST_UTIL_SOURCES
    tests/csv_reporter.c
    tests/test_utils.c
    tests/cten_tests.c
)

# Operator-specific tests
file(GLOB_RECURSE OPERATOR_TEST_SOURCES "tests/Operator/*.c")

# Gradient-specific tests (can be empty initially)
file(GLOB_RECURSE GRAD_TEST_SOURCES "tests/Grad/*.c")

# Combine all test sources
set(ALL_TEST_SOURCES
    ${TEST_UTIL_SOURCES}
    ${OPERATOR_TEST_SOURCES}
    ${GRAD_TEST_SOURCES}
)

# Create test executable with library sources and all test sources
add_executable(cten_tests ${ALL_TEST_SOURCES} ${LIB_SOURCES})

# Add MSVC-specific compiler options for tests
if(MSVC)
    target_compile_options(cten_tests PRIVATE /wd4305)
    target_compile_definitions(cten_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

target_compile_definitions(cten_tests PRIVATE "CTEN_BUILD_DIR_PATH=\"${CMAKE_BINARY_DIR}\"")

target_include_directories(cten_tests PRIVATE
    tests/Operator
    tests/Grad
)

set_target_properties(cten_tests PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Set output directory for the test executable
set_target_properties(cten_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Link math library for tests
if(NOT WIN32)
    target_link_libraries(cten_tests PRIVATE m)
endif()

# Enable testing
enable_testing()
add_test(NAME AllTests COMMAND cten_tests)

# Optional: Define a custom target to build and run tests easily
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS cten_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)