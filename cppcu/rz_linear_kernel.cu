#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <vector>
#include <stdio.h>
#include <algorithm>
#include <chrono>

#define MAX_GRID_SIZE  16384
#define SMLS 16  // BLOCK in tiled multiplication
#define TTILE 4 // per thread responsibility
#define SMLST 64 // SMLS * TTILE
#define SHIFT 3
__device__ int64_t hash_func(int64_t a, int64_t b, int64_t *  random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) % random_numbers[0];
}

__device__ int64_t hash_func3(int64_t a, int64_t b, int64_t c, int64_t * random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + c* random_numbers[1] + random_numbers[4]) % random_numbers[0];
}

inline __device__ int64_t location(int64_t i, int64_t j, int chunk_size, int64_t * random_numbers, int64_t range) {
    // we have chunked columwise for faster forward pass
    int64_t chunk_id = i / chunk_size;
    int64_t offset = i % chunk_size;
    return (hash_func(chunk_id, j, random_numbers) + offset) % range;
}

inline __device__ int64_t location_tile(int64_t tile_i, int64_t tile_j, int i, int j, int chunk_size,
                                        int64_t * random_numbers, int64_t range) {
    int id = j * SMLS + i;
    int chunk_id = id / chunk_size;
    int offset = id % chunk_size;
    return (hash_func3(tile_i, tile_j, chunk_id, random_numbers) + offset) % range;
}

//template<typename scalar_t>
__global__ void rz_linear_forward_cuda_kernel(
    float * weights, //  K x N (column major)
    float * input, // M x K (row major)
    float * output, // M x N (row major)
    int64_t * random_numbers,
    int M,
    int K,
    int N,
    int chunk_size
)
{
  int num_block_width = (N  + SMLST - 1) / SMLST;
  int total_number_of_blocks = (int)((M + SMLST - 1) / SMLST) * (int)((N + SMLST - 1) / SMLST) ;
  int block_idx = blockIdx.x;
  int block_x, block_y, tx, ty, gx, gy;

  __shared__ float shareI[SMLST][SMLST + SHIFT];
  __shared__ float shareM[SMLST][SMLST + SHIFT];
  float val[TTILE][TTILE] = {0};

#pragma unroll
  for (; block_idx < total_number_of_blocks; block_idx += gridDim.x) { // outer loop if the output matrix is too large
    block_x = block_idx / num_block_width;
    block_y = block_idx % num_block_width;
    tx = threadIdx.x; ty = threadIdx.y;
    gx = block_x * SMLST + tx;
    gy = block_y * SMLST + ty;

#pragma unroll
    for (int x_offset = 0; x_offset < TTILE; x_offset ++) {

#pragma unroll
      for (int y_offset = 0; y_offset < TTILE ; y_offset ++) {
        val[x_offset][y_offset] = 0.;
      }
    }
#pragma unroll
    for (int i = 0 ; i < (K + SMLST - 1) / SMLST ; i ++) {
      #pragma unroll
      for (int x_offset_abs = 0; x_offset_abs < SMLST ; x_offset_abs += SMLS) {
        #pragma unroll
        for (int y_offset_abs = 0; y_offset_abs < SMLST ; y_offset_abs += SMLS) {
          if (i*SMLST + ty + y_offset_abs < K && gx + x_offset_abs < M) {
            shareI[tx + x_offset_abs][ty + y_offset_abs] = input[(gx + x_offset_abs) * K  + i* SMLST + ty + y_offset_abs]; // row major (gx, i*SMLS+ty)
          } else {
            shareI[tx + x_offset_abs][ty + y_offset_abs] = 0.;
          }
          if (i*SMLST + tx + x_offset_abs < K && gy + y_offset_abs < N) {
            shareM[ty + y_offset_abs][tx + x_offset_abs]= weights[i* SMLST + (tx + x_offset_abs) + (gy + y_offset_abs) * K]; // coumn major (i*SMLS+tx, gy)
          } else {
            shareM[ty + y_offset_abs][tx + x_offset_abs]  = 0.;
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int x_offset = 0; x_offset < TTILE; x_offset ++) {

#pragma unroll
        for (int y_offset = 0; y_offset < TTILE ; y_offset ++) {

#pragma unroll
          for (int j = 0; j < SMLST; j ++ ) {
            val[x_offset][y_offset] += shareI[tx + x_offset*SMLS][j] * shareM[ty + y_offset*SMLS][j];
          }
        }
      }
      __syncthreads();
    }

#pragma unroll
    for (int x_offset = 0; x_offset < TTILE; x_offset ++) {

#pragma unroll
      for (int y_offset = 0; y_offset < TTILE ; y_offset ++) {
        if ((gx + x_offset * SMLS) < M && (gy + y_offset * SMLS) < N) {
          output[(gx + x_offset*SMLS)*N + (gy + y_offset * SMLS)] = val[x_offset][y_offset];
        }
      }
    }
  }
}

void rz_linear_forward_cuda (
    float * weights, // 1 x n
    float * input, // b x d1
    float * output,
    int64_t * random_numbers,
    int M,
    int K,
    int N,
    int chunk_size
    )
{
    dim3 block = dim3(SMLS, SMLS, 1);

    int total_number_of_blocks = (int)((M + SMLST - 1) / SMLST) * (int)((N + SMLST - 1) / SMLST) ;
    int grid = MAX_GRID_SIZE;
    if (total_number_of_blocks < MAX_GRID_SIZE) {
        grid = total_number_of_blocks;
    }

    //printf("problem config %d x %d, %d x %d \n", M,K,K,N);
    //printf("launching the kernel with grid:%d block:(%d, %d, %d) ||  total blocks:%d \n", grid, block.x, block.y, block.z, total_number_of_blocks);
    rz_linear_forward_cuda_kernel<<<grid, block>>>(
            weights,
            input,
            output,
            random_numbers,
            M,
            K,
            N,
            chunk_size
      );
}

void  rz_linear_forward(
    float *  weights,
    float *  input,
    float * output,
    int64_t * random_numbers,
    int M,
    int K,
    int N,
    int chunk_size
)
{
    return rz_linear_forward_cuda(weights, input, output, random_numbers, M, K, N, chunk_size);
}

void verify(float * weights, float * input, float * output, int64_t * random_numbers, int M, int K, int N,  int chunk_size) {
    // cpp verify the matrix mulitplication
    float val;
    for (int i = 0 ; i < M; i++) {
        for (int j=0; j < N; j++) {
            val = 0;
            for (int k=0; k < K;k++) {
                val += input[i*K + k]  * weights[k + j * K];
            }
            if ( abs(output[i*N + j] -  val)  > 1e-5 * val) {
                printf("mismatch @ (%d,%d) expected : %f got: %f\n", i, j, val, output[i*N+j]);
            }
        }
    }
}

void init_random( float * A, int size) {
    for (int i=0; i < size; i ++) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void printM(float * A, int m, int n, bool rmajor) {
    for(int i=0; i < m; i++) {
        for (int j=0; j < n; j++) {
            if ( rmajor ) {
                printf(" %f", A[i*n + j]);
            } else {
                printf(" %f", A[i + j*m]);
            }
        }
        printf("\n");
    }
}

int main() {
  auto t_start  = std::chrono::high_resolution_clock::now();
  auto t_end  = std::chrono::high_resolution_clock::now();

  int M = 1024;
  int K = 1024;
  int N = 102400;
  int chunk_size = 2;
  printf("problem size : %d x %d x %d\n", M, K, N);
  
  float * h_I, * h_W, * h_O;
  
  h_I = (float * ) malloc(M * K * sizeof(float));
  h_W = (float * ) malloc(N * K * sizeof(float));
  h_O = (float * ) malloc(M * N * sizeof(float));

  float * d_I, * d_W, * d_O;
  cudaMalloc((void **) & d_I, M * K * sizeof(float));
  cudaMalloc((void **) & d_W, N * K * sizeof(float));
  cudaMalloc((void **) & d_O, M * N * sizeof(float));

  // initialize h_I and h_W;
  srand(1);
  init_random(h_I, M*K);
  init_random(h_W, K*N);

  // copy to cuda
  cudaMemcpy(d_I, h_I, M*K*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_W, h_W, N*K*sizeof(float), cudaMemcpyHostToDevice);
  int ITR = 100;
  t_start = std::chrono::high_resolution_clock::now();
  for(int p=0;p < ITR; p ++) {
    rz_linear_forward_cuda(d_W, d_I, d_O, NULL, M, K, N, chunk_size);
    cudaDeviceSynchronize();
  }
  t_end = std::chrono::high_resolution_clock::now();
  cudaMemcpy(h_O, d_O, M*N*sizeof(float), cudaMemcpyDeviceToHost);
  printf("time taken by our call smls: %d smlst: %d grid: %d is %f (ms)\n", SMLS, SMLST, MAX_GRID_SIZE,
                                std::chrono::duration<double, std::milli>(t_end - t_start).count() / ITR);
  //printf("verifying..\n");
  //verify(h_W, h_I, h_O, NULL, M, K, N, chunk_size);
  //printf("input\n");
  //printM(h_I, M, K, true);

  //printf("weight\n");
  //printM(h_W, K, N, false);

  //printf("output\n");
  //printM(h_O, M, N, true);
   
  free(h_I);
  free(h_W);
  free(h_O);
  cudaFree(d_I);
  cudaFree(d_W);
  cudaFree(d_O);
  return 0;
}
