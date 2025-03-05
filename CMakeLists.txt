cmake_minimum_required(VERSION 3.10)
project(cTensor LANGUAGES C)

# Include project headers
include_directories(include)

# Collect all source files
file(GLOB_RECURSE SOURCES
    "src/*.c"
    "src2/*.c"
)

# Create executable
add_executable(cten_exe ${SOURCES})

# Set C standard
set_target_properties(cten_exe PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

# Link math library only on non-Windows systems
if(NOT WIN32)
    target_link_libraries(cten_exe PRIVATE m)
endif()