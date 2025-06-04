FROM ubuntu:22.04

# Set non-interactive frontend to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    libomp-dev \
    cpuinfo \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Reset frontend
ENV DEBIAN_FRONTEND=

# Set working directory
WORKDIR /app

# Copy benchmark source code
COPY benchmark.cpp .
COPY CMakeLists.txt .

# Build the benchmark
RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Set environment variables for OpenBLAS (allow multi-threading)
ENV OPENBLAS_NUM_THREADS=16
ENV OMP_NUM_THREADS=16

# Run the benchmark by default
CMD ["./build/flops_benchmark"]