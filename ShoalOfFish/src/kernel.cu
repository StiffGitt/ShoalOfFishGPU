#include "cuda_runtime.h"
#include <cstdio>
#include "kernel.cuh"
#include "Fish.h"
#include <thrust>

#define BLOCKS 256
#define THREADS 1024

Fish *dev_fishes;
Fish *dev_gathered_fishes;
int *dev_indices;
int *dev_grid_first;
int *dev_grid_last;
int *dev_cell_idx;
int N;

// global
void assign_grid(Fish *fishes, float cell_size, int* cell_idx, int* indices)
{
    int grid_size = (int)(2.0f / cell_size) + 1;
    for (int i = 0; i < N; i++)
    {
        float x = fishes[i].x + 1.0f;
        float y = fishes[i].y + 1.0f;
        int r = (int)(y / cell_size);
        int c = (int)(x / cell_size);
        dev_cell_idx[i] = r * grid_size + c;
        indices[i] = i;
    }
}

void make_calculations_cuda(Fish *fishes, float r1, float r2, float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode)
{
    cudaError_t cudaStatus;
    
    float cell_size = r2 * 2;
    int grid_size = (int)(2.0f / cell_size) + 1;
    int grid_length = (grid_size) * (grid_size);

    cudaStatus = cudaMemcpy(dev_fishes, fishes, N * sizeof(Fish), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    cudaStatus = cudaMemset(dev_grid_first, -1, grid_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
    }
    cudaStatus = cudaMemset(dev_grid_last, -1, grid_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed!");
    }

    assign_grid(dev_fishes, cell_size, dev_cell_idx, dev_indices);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }

    auto fishes_pointer = thrust::device_pointer_cast(dev_fishes);
    auto gathered_fishes_pointer = thrust::device_pointer_cast(dev_gathered_fishes);
    auto cell_idx_pointer = thrust::device_pointer_cast(dev_cell_idx);
    auto indices_pointer = thrust::device_pointer_cast(dev_indices);


}

void init_cuda(int n, int grid_length)
{
    cudaError_t cudaStatus;

    N = n;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
    }
    cudaStatus = cudaMalloc((void**)&dev_fishes, N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_gathered_fishes, N * sizeof(Fish));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_indices, N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_grid_first, grid_length * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_grid_last, grid_length * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_cell_idx, N * sizeof(unsigned int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
}

void free_cuda()
{
    cudaFree(dev_fishes);
    cudaFree(dev_gathered_fishes);
    cudaFree(dev_indices);
    cudaFree(dev_grid_first);
    cudaFree(dev_grid_last);
    cudaFree(dev_cell_idx);
}