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

# Link math library (cross-platform)
if(NOT WIN32)
    target_link_libraries(cten_exe PRIVATE m)
endif()

# Testing setup
file(GLOB TEST_SOURCES "tests/*.c")

# Create test executable with library sources
add_executable(cten_tests ${TEST_SOURCES} ${LIB_SOURCES})

# Include tests directory for test_utils.h
target_include_directories(cten_tests PRIVATE tests)

# Set C standard for tests
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
add_test(NAME OperatorTests COMMAND cten_tests)