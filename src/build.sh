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
cp -r include/ ${CTR_BUILD_DIR}/
nvcc -O3 obj_loader.cpp -o ${CTR_BUILD_DIR}/obj_loader
g++ -o ${CTR_BUILD_DIR}/fbx_loader fbx_loader.cpp include/ufbx.c -I${CTR_BUILD_DIR}/include
