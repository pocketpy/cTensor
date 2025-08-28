#!/bin/sh

set -e
mkdir -p build
cd build
cmake ..
cmake --build .

echo "\nBuild complete. Ctensor Tests Executable is in the 'build/bin' directory. Run 'build/bin/cten_tests' to run the tests. \n"
echo "\nFor running the main(Demo) executable, run 'build/cten_exe' \n"