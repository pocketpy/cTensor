set -e

SRC=$(find src/ -name "*.c")

FLAGS="-std=c11 -lm -Iinclude -O0 -Wfatal-errors -g -DDEBUG"

SANITIZE_FLAGS="-fsanitize=address,leak,undefined"

if [ "$(uname)" == "Darwin" ]; then
    SANITIZE_FLAGS="-fsanitize=address,undefined"
fi

echo "Compiling C files..."
clang $FLAGS $SANITIZE_FLAGS $SRC src2/main.c -o main

