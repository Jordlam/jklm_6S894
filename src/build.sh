#!/bin/bash

# Output build directory path.
CTR_BUILD_DIR=/build

echo "Building the project..."

# ----------------------------------------------------------------------------------
# ------------------------ PUT YOUR BULDING COMMAND(s) HERE ------------------------
# ----------------------------------------------------------------------------------
# ----- This sctipt is executed inside the development container:
# -----     * the current workdir contains all files from your src/
# -----     * all output files (e.g. generated binaries, test inputs, etc.) must be places into $CTR_BUILD_DIR
# ----------------------------------------------------------------------------------
# Copy tests.
# cp jax_test.py ${CTR_BUILD_DIR}/
# cp torch_test.py ${CTR_BUILD_DIR}/
# cp arch_test.py ${CTR_BUILD_DIR}/

# Build code.
nvcc -O3 vector_add.cu -o ${CTR_BUILD_DIR}/vector_add

# Other files
cp -r data/ ${CTR_BUILD_DIR}/

# Build helpers
HELPER_DIR=include/
g++ -O3 -c -o ${CTR_BUILD_DIR}/ufbx.o ${HELPER_DIR}ufbx.c
g++ -O3 -c -o ${CTR_BUILD_DIR}/fbx_loader.o ${HELPER_DIR}fbx_loader.cpp
g++ -O3 -c -o ${CTR_BUILD_DIR}/obj_loader.o ${HELPER_DIR}obj_loader.cpp
g++ -O3 -c -o ${CTR_BUILD_DIR}/tgaimage.o ${HELPER_DIR}tgaimage.cpp

# Main files
g++ -O3 -o ${CTR_BUILD_DIR}/cpu_rasterizer cpu_rasterizer.cpp ${CTR_BUILD_DIR}/ufbx.o ${CTR_BUILD_DIR}/fbx_loader.o ${CTR_BUILD_DIR}/obj_loader.o ${CTR_BUILD_DIR}/tgaimage.o
