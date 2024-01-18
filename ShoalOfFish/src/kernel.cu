#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdio>
#include "kernel.cuh"
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/gather.h>
#include "Consts.h"

#define BLOCKS 256
#define THREADS 1024

Fish *dev_fishes;
Fish *dev_gathered_fishes;
int *dev_indices;
int *dev_grid_first;
int *dev_grid_last;
int *dev_cell_idx;
float *dev_v_x;
float *dev_v_y;

__global__ void assign_grid(Fish *fishes, float cell_size, int* cell_idx, int* indices)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int grid_size = (int)(2.0f / cell_size) + 1;
    float x = fishes[idx].x + 1.0f;
    float y = fishes[idx].y + 1.0f;
    int r = (int)(y / cell_size);
    int c = (int)(x / cell_size);
    cell_idx[idx] = r * grid_size + c;
    indices[idx] = idx;
}

__global__ void find_border_cells(int* grid_first, int* grid_last, int* cell_idx)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    if (idx == 0)
    {
        grid_first[cell_idx[0]] = 0;
        return;
    }
    int cur_cell = cell_idx[idx];
    int prev_cell = cell_idx[idx - 1];
    if (cur_cell != prev_cell)
    {
        grid_last[prev_cell] = idx;
        grid_first[cur_cell] = idx;
    }
    if (idx == N - 1)
        grid_last[cur_cell] = N;
}

/*float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode,*/

__global__ void calculate_v(Fish* gathered_fishes, int* grid_first, int* grid_last, int* cell_idx, float* v_x, float* v_y, float r1, float r2, int grid_size, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef, float preyCoef, bool predatorMode)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    int grid_length = (grid_size) * (grid_size);
    float r1sq = r1 * r1;
    float r2sq = r2 * r2;

    float x = gathered_fishes[idx].x;
    float y = gathered_fishes[idx].y;
    float vx = gathered_fishes[idx].dx;
    float vy = gathered_fishes[idx].dy;
    float cumX = 0.0, cumY = 0.0, cumVx = 0.0, cumVy = 0.0, visibleFriendlyCount = 0.0, visiblePreyCount = 0.0,
        closestPredatorX = -1.0, closestPredatorY = -1.0f, closeDx = 0.0, closeDy = 0.0, cumXP = 0.0, cumYP = 0.0,
        closestPredatorDsq = 8.0f;
    int cell = cell_idx[idx];
    int cells_to_check[] = { cell - 1, cell, cell + 1,
        cell - grid_size - 1, cell - grid_size, cell - grid_size + 1,
        cell - grid_size + 1, cell + grid_size, cell + grid_size + 1 };
    
    //for(int cidx = 0; cidx < 9; cidx++)
    //{
    //    int nc = cells_to_check[cidx];
    //    if (nc < 0 || nc > grid_length || grid_first[nc] < 0)
    //        /*continue*/;
    //    //for (int j = grid_first[nc]; j < grid_last[nc]; j++)
    //    //{
    //    //    if (j == idx)
    //    //        continue;
    //    //    float xj = gathered_fishes[j].x;
    //    //    float yj = gathered_fishes[j].y;
    //    //    float dx = x - xj;
    //    //    float dy = y - yj;

    //    //    if (fabsf(dx) < r2 && fabsf(dy) < r2)
    //    //    {
    //    //        float dsq = dx * dx + dy * dy;
    //    //        if (dsq < r2)
    //    //        {
    //    //            // Avoid predators
    //    //            if (gathered_fishes[idx].species < gathered_fishes[j].species)
    //    //            {
    //    //                if (closestPredatorDsq > dsq)
    //    //                {
    //    //                    closestPredatorDsq = dsq;
    //    //                    closestPredatorX = xj;
    //    //                    closestPredatorY = yj;
    //    //                }
    //    //            }
    //    //            // Hunt prey
    //    //            if (gathered_fishes[idx].species > gathered_fishes[j].species)
    //    //            {
    //    //                visiblePreyCount++;
    //    //                cumXP += xj;
    //    //                cumYP += yj;
    //    //            }
    //    //            if (dsq < r1sq)
    //    //            {
    //    //                // Separation
    //    //                closeDx += (x - xj); /** (1 - (dx / r1));*/
    //    //                closeDy += (y - yj); /** (1 - (dy / r1));*/
    //    //            }
    //    //            else
    //    //            {
    //    //                if (gathered_fishes[idx].species == gathered_fishes[j].species && gathered_fishes[idx].species <= 1)
    //    //                {
    //    //                    visibleFriendlyCount++;
    //    //                    // Alignment
    //    //                    cumVx += gathered_fishes[j].dx;
    //    //                    cumVy += gathered_fishes[j].dy;

    //    //                    // Cohension
    //    //                    cumX += xj;
    //    //                    cumY += yj;
    //    //                }
    //    //            }
    //    //        }
    //    //    }
    //    //}
    //}
    
    // Avoid predators
    if (predatorMode && closestPredatorDsq < r2)
    {
        vx += (x - closestPredatorX) * predatorsCoef;
        vy += (y - closestPredatorY) * predatorsCoef;
    }

    //// Chase prey
    /*if (predatorMode && visiblePreyCount > 0)
    {
        vx += ((cumXP / visiblePreyCount) - x) * preyCoef;
        vy += ((cumYP / visiblePreyCount) - y) * preyCoef;
    }*/

    gathered_fishes[idx].x = 40.0f;
    //// Separation
    /*vx += closeDx * avoidCoef;
    vy += closeDy * avoidCoef;*/

    //if (visibleFriendlyCount > 0)
    //{
    //    // Alignment
    //    vx += ((cumVx / visibleFriendlyCount) - gathered_fishes[idx].dx) * alignCoef;
    //    vy += ((cumVy / visibleFriendlyCount) - gathered_fishes[idx].dy) * alignCoef;

    //    // Cohension
    //    vx += ((cumX / visibleFriendlyCount) - x) * cohensionCoef;
    //    vy += ((cumY / visibleFriendlyCount) - y) * cohensionCoef;
    //}
}

__global__ void scale_v(Fish* gathered_fishes, float* v_x, float* v_y, float maxV, float minV, float curX, float curY, float curActive)
{
    //// Turn from edges
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float x = gathered_fishes[idx].x;
    float y = gathered_fishes[idx].y;
    float vx = v_x[idx];
    float vy = v_y[idx];
    bool isTurning = false;
    if (x < LEFT_EDGE && vx < minV)
    {
        isTurning = true;
        if (x < -1.0 + MARGIN)
            vx = -vx;
        else
            vx += TURN_COEF + (vx * vx) / (x + 1.0f);
    }
    if (x > RIGHT_EDGE && vx > -minV)
    {
        isTurning = true;
        if (x > 1.0 - MARGIN)
            vx = -vx;
        else
            vx -= TURN_COEF + (vx * vx) / (1.0f - x);
    }

    if (y < BOTTOM_EDGE && vy < minV)
    {
        isTurning = true;
        if (y < -1.0 + MARGIN)
            vy = -vy;
        else
            vy += TURN_COEF + (vy * vy) / (y + 1.0f);
    }
    if (y > TOP_EDGE && vy > -minV)
    {
        isTurning = true;
        if (y > 1.0 - MARGIN)
            vy = -vy;
        else
            vy -= TURN_COEF + (vy * vy) / (1.0f - y);
    }

    float dcx = x - curX;
    float dcy = y - curY;
    if (curActive && dcx * dcx + dcy * dcy < CURSOR_RANGE * CURSOR_RANGE)
    {
        vx += dcx * CURSOR_COEF;
        vy += dcy * CURSOR_COEF;
    }

    //// Adjust velocity to minmax
    float v = sqrtf(vx * vx + vy * vy);
    if (v < minV && !isTurning)
    {
        vx = (vx / v) * minV;
        vy = (vy / v) * minV;
    }
    else if (v > maxV)
    {
        vx = (vx / v) * maxV;
        vy = (vy / v) * maxV;
    }

    v_x[idx] = vx;
    v_y[idx] = vy;
    gathered_fishes[idx].dx = vx;
    gathered_fishes[idx].dx = 50.0f;
    gathered_fishes[idx].dy = vy;

    gathered_fishes[idx].x = 0;
    gathered_fishes[idx].y = y + vy;
}

__global__ void move_fishes(Fish *gathered_fishes,float* v_x, float* v_y)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    gathered_fishes[idx].dx = v_x[idx];
    gathered_fishes[idx].dy = v_y[idx];

    gathered_fishes[idx].x += v_x[idx];
    gathered_fishes[idx].y += v_y[idx];
}

void make_calculations_cuda(Fish *fishes, float r1, float r2, float turnCoef, float cohensionCoef, float avoidCoef, float alignCoef, float predatorsCoef,
    float preyCoef, float maxV, float minV, float curX, float curY, float curActive, bool predatorMode)
{
    cudaError_t cudaStatus;
    
    float cell_size = r2 * 2;
    int grid_size = (int)(2.0f / cell_size) + 1;
    int grid_length = (grid_size) * (grid_size);
    int* tab = (int*)malloc(grid_length * sizeof(int));

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

    assign_grid << < BLOCKS, THREADS >> > (dev_fishes, cell_size, dev_cell_idx, dev_indices);

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }
    
    auto cell_idx_pointer = thrust::device_pointer_cast(dev_cell_idx);
    auto indices_pointer = thrust::device_pointer_cast(dev_indices);
    thrust::sort_by_key(cell_idx_pointer, cell_idx_pointer + N, indices_pointer);

    find_border_cells << < BLOCKS, THREADS >> > (dev_grid_first, dev_grid_last, dev_cell_idx);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }

    auto fishes_pointer = thrust::device_pointer_cast(dev_fishes);
    auto gathered_fishes_pointer = thrust::device_pointer_cast(dev_gathered_fishes);
    thrust::gather(indices_pointer, indices_pointer + N, fishes_pointer, gathered_fishes_pointer);

    calculate_v << < BLOCKS, THREADS >> > (dev_gathered_fishes, dev_grid_first, dev_grid_last, dev_cell_idx, dev_v_x, dev_v_y, r1, r2, grid_size, cohensionCoef, avoidCoef, alignCoef, predatorsCoef, preyCoef, predatorMode);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }

    /*cudaStatus = cudaMemcpy(fishes, dev_gathered_fishes, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    for (int i = 0; i < N; i++)
        std::cout << fishes[i].x << ", " << fishes[i].y << ": " << fishes[i].dx << ", " << fishes[i].dy << std::endl;*/

    //move_fishes << < BLOCKS, THREADS >> > (dev_gathered_fishes, dev_v_x, dev_v_y);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
    }

    /*cudaStatus = cudaMemcpy(tab, dev_grid_last, grid_length * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    for (int i = 0; i < grid_length; i++)
        std::cout << tab[i] << std::endl;*/
    cudaStatus = cudaMemcpy(fishes, dev_gathered_fishes, N * sizeof(Fish), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
    }
    for (int i = 0; i < N; i++)
        std::cout << fishes[i].x << ", " << fishes[i].y << ": " << fishes[i].dx << ", " << fishes[i].dy << std::endl;

}

void init_cuda(int grid_length)
{
    cudaError_t cudaStatus;

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
    cudaStatus = cudaMalloc((void**)&dev_indices, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_grid_first, grid_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_grid_last, grid_length * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_cell_idx, N * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_v_x, N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
    }
    cudaStatus = cudaMalloc((void**)&dev_v_y, N * sizeof(float));
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
    cudaFree(dev_v_x);
    cudaFree(dev_v_y);
}