#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

__global__ void pairwiseAcceleration(vector3* accels, double* hPos, double* mass, int numEntities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < numEntities && j < numEntities) {
        if (i == j) {
            FILL_VECTOR(accels[i*numEntities + j], 0, 0, 0);
        }
        else {
            vector3 distance;
            for (int k = 0; k < 3; k++) {
                distance[k] = hPos[i*3 + k] - hPos[j*3 + k];
            }
            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = -1 * GRAV_CONSTANT * mass[j] / magnitude_sq;
            FILL_VECTOR(accels[i*numEntities + j], accelmag*distance[0] / magnitude, accelmag*distance[1] / magnitude, accelmag*distance[2] / magnitude);
        }
    }
}

void compute() {
    // Allocate memory on the host
    vector3* values = (vector3*)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);

    // Allocate memory on the GPU
    vector3* dAccels;
    double* dPos;
    double* dMass;
    int size = NUMENTITIES * NUMENTITIES * sizeof(vector3);
    cudaMalloc((void**)&dAccels, size);
    cudaMalloc((void**)&dPos, NUMENTITIES * 3 * sizeof(double));
    cudaMalloc((void**)&dMass, NUMENTITIES * sizeof(double));

    // Copy data from host to device
    cudaMemcpy(dPos, hPos, NUMENTITIES * 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dMass, mass, NUMENTITIES * sizeof(double), cudaMemcpyHostToDevice);

    // Launch kernel for pairwise acceleration computation
    dim3 blockDim(16, 16);
    dim3 gridDim((NUMENTITIES + blockDim.x - 1) / blockDim.x, (NUMENTITIES + blockDim.y - 1) / blockDim.y);
    pairwiseAcceleration<<<gridDim, blockDim>>>(dAccels, dPos, dMass, NUMENTITIES);
    cudaDeviceSynchronize();

    // Copy result back from device to host
    cudaMemcpy(values, dAccels, size, cudaMemcpyDeviceToHost);

    // Free memory on the GPU
    cudaFree(dAccels);
    cudaFree(dPos);
    cudaFree(dMass);

    // Perform calculations using the computed accelerations
    for (int i = 0; i < NUMENTITIES; i++) {
        vector3 accel_sum = { 0, 0, 0 };
        for (int j = 0; j < NUMENTITIES; j++) {
            for (int k = 0; k < 3; k++) {
                accel_sum[k] += values[i * NUMENTITIES + j][k];
            }
        }

                // Update velocity and position
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }

    // Free memory on the host
    free(values);
}

