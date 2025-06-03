#ifndef TEST_CONFIG_H
#define TEST_CONFIG_H

#define TEST_FLOAT_TOLERANCE 1e-6

#if defined(_WIN32) || defined(_WIN64)
    #define PLATFORM_NAME "windows"
#elif defined(__linux__)
    #define PLATFORM_NAME "linux"
#elif defined(__APPLE__) || defined(__MACH__)
    #define PLATFORM_NAME "macos"
#else
    #define PLATFORM_NAME "unknown"
#endif

#define CTENSOR_MAX_DIMS 4

#endif
