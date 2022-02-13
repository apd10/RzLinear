#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorAccessor.h>

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

#define MAX_GRID_SIZE  8192
#define MAX_BLOCK_SIZE  128

#define SMLS 16  // BLOCK in tiled multiplication
#define SMLSMASK 15L
#define TTILE 4 // per thread responsibility TTILE x TTILE
#define SMLST 64 // SMLS * TTILE
#define SHIFT 1

#define BITMASK 1048575L  // 2^20 -1 for cases when range is < 10^6

#define BASIC 0
#define TILED 1

__device__ int64_t hash_func(int64_t a, int64_t b, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) % random_numbers[0]; // modulo with  large numbers is expensive
    //return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) & BITMASK; // TODO
}

__device__ int64_t hash_func3(int64_t a, int64_t b, int64_t c, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + c* random_numbers[1] + random_numbers[4]) % random_numbers[0];
}

__device__ int64_t hash_func4(int64_t a, int64_t b, int64_t c, int64_t d, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + c* random_numbers[1] + d * random_numbers[4] + random_numbers[5]) % random_numbers[0];
    //return (a * random_numbers[3] + b * random_numbers[2] + c* random_numbers[1] + d * random_numbers[4] + random_numbers[5]) & (int64_t) BITMASK;
}

inline __device__ int64_t location(int64_t i, int64_t j, int chunk_size, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers, int64_t range) {
    // we have chunked columwise for faster forward pass
    int64_t chunk_id = i / chunk_size;
    int64_t offset = i % chunk_size;
    return (hash_func(chunk_id, j, random_numbers) + offset) % range;
}


inline __device__ int64_t location_tiled(int64_t i, int64_t j, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers, int64_t range) {
    // we have chunked columwise for faster forward pass
    int64_t block_x = i / SMLS;
    int64_t block_y = j / SMLS; 
    int64_t ix = i & SMLSMASK;
    int64_t iy = j & SMLSMASK;
    int64_t loc = (hash_func(block_x, block_y, random_numbers)) % (range - SMLS * SMLS + 1) + ix * SMLS + iy;
    return loc;
}

inline __device__ int64_t location_tile(int64_t tile_i, int64_t tile_j, int i, int j, int chunk_size,
                                        const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers, int64_t range) {
    int id = j * SMLS + i;
    int chunk_id = id / chunk_size;
    int offset = id % chunk_size;
    return (hash_func3(tile_i, tile_j, chunk_id, random_numbers) + offset) % range;
}

template<typename scalar_t>
__global__ void rz_linear_forward_cuda_kernel(
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  hashed_weights,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  output,
            torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
            int batch,
            int input_dim,
            int output_dim,
            int chunk_size,
            int hashed_weight_size
)
{
  int out_x = blockIdx.x;
  int out_y = threadIdx.y;
  scalar_t val = 0;
  int num_chunks = (input_dim + chunk_size - 1)/ chunk_size;
  int idx = 0;
  int kidx =0;
  for (; out_x < batch; out_x+= gridDim.x) {
    for(; out_y < output_dim; out_y += blockDim.y) {
      val = 0;
      for(int c = 0; c < num_chunks;c ++) {
        idx = hash_func(c, out_y, random_numbers) % (hashed_weight_size); // 
        for( int ic = 0; ic < chunk_size ; ic ++) {
            kidx = c * chunk_size + ic;
            if (kidx < input_dim) {
                val+= input[out_x][kidx] * hashed_weights[idx];
                idx  = (idx + 1) % hashed_weight_size;
            }
        }
      }
      output[out_x][out_y] = val;
    }
  }
}

template<typename scalar_t>
__global__ void rz_linear_backward_cuda_kernel_input(
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  hashed_weights,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  out_grad,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input_grad,
            torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
            int batch,
            int input_dim,
            int output_dim,
            int chunk_size,
            int64_t hashed_weight_size,
            bool tiled
)
{
  int in_x = blockIdx.x;
  int in_y = threadIdx.y;
  scalar_t val = 0;
  int num_chunks = (input_dim + chunk_size - 1)/ chunk_size;
  int64_t idx = 0;

  
  #pragma unroll
  for (; in_x < batch; in_x+= gridDim.x) {

  #pragma unroll
    for(; in_y < input_dim; in_y += blockDim.y) {

      #pragma unroll
       for(int k=0; k< output_dim;k++) {
         if (tiled) {
            idx = location_tiled(in_y, k, random_numbers, hashed_weight_size);
            //if (in_y % 16 ==0 && k % 16 == 0)
            //    printf("[i ]location_tiled: %ld %ld %ld\n", (int64_t) in_y, (int64_t) k, idx);
         }
         else {      
            idx = location(in_y, k, chunk_size, random_numbers, hashed_weight_size);
         }
          input_grad[in_x][in_y] += hashed_weights[idx] * out_grad[in_x][k];
       }
    }
  }
}


template<typename scalar_t>
__global__ void rz_linear_backward_cuda_kernel_weight(
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  hashed_weights,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  out_grad,
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  weight_grad,
            torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
            int batch,
            int input_dim,
            int output_dim,
            int chunk_size,
            int64_t hashed_weight_size,
            bool tiled
)
{
  int wt_x = blockIdx.x;
  int wt_y = threadIdx.y;
  scalar_t val = 0;
  int num_chunks = (input_dim + chunk_size - 1)/ chunk_size;
  int64_t loc = 0;
  //printf("%d %d (%d, %d)\n", wt_x, wt_y, input_dim, output_dim);
  
  #pragma unroll
  for (; wt_x < input_dim; wt_x+= gridDim.x) {

    #pragma unroll
    for(; wt_y < output_dim; wt_y += blockDim.y) {
        val = 0;

        #pragma unroll
        for(int k=0;k< batch;k++) {
            val += input[k][wt_x] * out_grad[k][wt_y];
        }
        // multiple threads will write to this.
        if (tiled) {
          loc = location_tiled(wt_x, wt_y, random_numbers, hashed_weight_size);

            //if (wt_x % 16 ==0 && wt_y % 16 == 0)
            //printf("[wt]location_tiled: %ld %ld %ld\n", (int64_t)wt_x, (int64_t)wt_y, loc);
        } else {
          loc = location(wt_x, wt_y, chunk_size, random_numbers, hashed_weight_size);
        }

        atomicAdd(& weight_grad[loc], val);
        //printf("%d %d %d adding %.4f\n", wt_x, wt_y, loc, val);
    }
  }
}


torch::Tensor rz_linear_forward_cuda (
    const torch::Tensor& hashed_weights, // 1 x n
    const torch::Tensor& input, // b x d1
    const torch::Tensor& random_numbers,
    int input_dim,
    int output_dim,
    int chunk_size
    )
{
    
    int64_t hashedWeightSize = hashed_weights.size(0);
    auto output = at::empty({input.size(0), output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    int x_max = input.size(0);
    int y_max = output_dim;

    dim3 block = dim3(1, MAX_BLOCK_SIZE, 1);
    if (y_max < MAX_BLOCK_SIZE) {
        block = dim3(1, y_max, 1);
    }
    
    dim3 grid = dim3(MAX_GRID_SIZE, 1, 1);
    if ( x_max < MAX_GRID_SIZE) {
        grid = dim3(x_max, 1, 1);
    }

    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "rz_linear_forward_cuda", ([&] {
        rz_linear_forward_cuda_kernel<scalar_t><<<grid, block, 0, stream>>>(
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            input.size(0),
            input_dim,
            output_dim,
            chunk_size,
            hashed_weights.size(0)
      );
    }));
   return output;
}


template<typename scalar_t>
__global__ void rz_linear_forward_cuda_kernel_tiled(
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  hashed_weights,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  output,
            torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
            int M,
            int K,
            int N,
            int chunk_size,
            int hashed_weight_size
)
{
  int num_block_width = (N  + SMLST - 1) / SMLST;
  int total_number_of_blocks = (int)((M + SMLST - 1) / SMLST) * (int)((N + SMLST - 1) / SMLST) ;
  int block_idx = blockIdx.x;
  int block_x, block_y, tx, ty, gx, gy;
  int64_t idx;

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
          // SMLSxSMLS = (block_x, block_y, x_offset_abs/SMLS, y_offset_abs / SMLS)
          idx = location_tiled(i*SMLST + x_offset_abs, block_y * SMLST + y_offset_abs, random_numbers, hashed_weight_size);
          //if ( tx == 0 && ty == 0) {
          //    printf("location_tiled: %ld %ld %ld\n", (int64_t)(i*SMLST + x_offset_abs),(int64_t)( block_y * SMLST + y_offset_abs), idx);
          //}
          if (i*SMLST + ty + y_offset_abs < K && gx + x_offset_abs < M) {
            shareI[tx + x_offset_abs][ty + y_offset_abs] = input[(gx + x_offset_abs)][i* SMLST + ty + y_offset_abs]; // row major (gx, i*SMLS+ty)
          } else {
            shareI[tx + x_offset_abs][ty + y_offset_abs] = 0.;
          }
          if (i*SMLST + tx + x_offset_abs < K && gy + y_offset_abs < N) {
            //shareM[ty + y_offset_abs][tx + x_offset_abs]= weights[i* SMLST + (tx + x_offset_abs) + (gy + y_offset_abs) * K]; // coumn major (i*SMLS+tx, gy)

            shareM[ty + y_offset_abs][tx + x_offset_abs]= hashed_weights[idx + tx * SMLS + ty]; // coumn major (i*SMLS+tx, gy)
            //shareM[ty + y_offset_abs][tx + x_offset_abs]= hashed_weights[location_tiled(i*SMLST + (tx + x_offset_abs), gy + y_offset_abs, random_numbers, hashed_weight_size)];
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
          output[(gx + x_offset*SMLS)][(gy + y_offset * SMLS)] = val[x_offset][y_offset];
        }
      }
    }
  }
}


template<typename scalar_t>
__global__ void rz_linear_forward_cuda_kernelXXX(
            torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>  hashed_weights,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  input,
            torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>  output,
            torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
            int batch,
            int input_dim,
            int output_dim,
            int chunk_size,
            int hashed_weight_size
)
{
  /* strategy is that each block is responsible for SIDE x SIDE chunk of output. */
  int total_output_blocks_x = ((batch + SMLS - 1) / SMLS);
  int total_output_blocks_y = ((output_dim + SMLS - 1)/  SMLS);
  int total_interim_blocks_k = ((input_dim + SMLS - 1)/  SMLS);
  int total_output_blocks = total_output_blocks_x * total_output_blocks_y;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int block_x;
  int block_y;
  int i_block_x;
  int i_block_y;
  int loc;
  int r,ir,ix,iy;

  scalar_t val;

  __shared__ scalar_t local_output [SMLS][SMLS]; // TODO +1 is the padding so that two rows do not belong to exaclty same memory banks. 
  __shared__ scalar_t local_input [SMLS][SMLS];
  __shared__ scalar_t local_weights [SMLS][SMLS];
  
  for (int oblock = bid; oblock < total_output_blocks; oblock += gridDim.x) {
      block_x = oblock % total_output_blocks_x;
      block_y = oblock / total_output_blocks_x;

      // set hte local_output to 0
      for( int itid = tid; itid < SMLS * SMLS; itid += blockDim.x) {
        // keep warp in row
        i_block_x = itid / SMLS;
        i_block_y = itid % SMLS;
        local_output[i_block_x][i_block_y] = 0;
      }

      // block_x, block_y is the block coordinates
      for (int interim = 0; interim < total_interim_blocks_k; interim ++ ) {
          /* we will now load the input chunk of size SIDE x SIDE into local memory*/
          // copy block_x, interim from input   //  local_input
          for( int itid = tid; itid < SMLS * SMLS; itid += blockDim.x) {
              // keep warp in row
              i_block_x = itid / SMLS;
              i_block_y = itid % SMLS;
              if (block_x * SMLS + i_block_x < batch && interim * SMLS  + i_block_y < input_dim) {
                  local_input[i_block_x][i_block_y] = input[block_x * SMLS + i_block_x][interim * SMLS  + i_block_y];
              } else {
                  local_input[i_block_x][i_block_y] = 0;
              }
          }

          // copy interim, block_y from weight
          // we will hash the SxS tile coordinates  // local_weights
          for( int itid = tid; itid < SMLS * SMLS; itid += blockDim.x) {
              i_block_x = itid % SMLS;
              i_block_y = itid / SMLS;
              if (interim * SMLS + i_block_x < input_dim && block_y * SMLS  + i_block_y < output_dim) {
                  // stored in column major order
                  loc = location_tile(interim, block_y, i_block_x, i_block_y, chunk_size, random_numbers, hashed_weight_size);
                  local_weights[i_block_y][i_block_x] = hashed_weights[loc];
              } else {
                  local_weights[i_block_y][i_block_x] = 0;
              }
          }

          // local matrix multiplication now
          for( int itid = tid; itid < SMLS * SMLS; itid += blockDim.x) {
              r = itid/SMLS;
              ir = itid % SMLS; 
              ix = (ir <= r) ? r - ir : (SMLS - (ir - r));
              iy = ir;
              // ix,iy is the local_output we will compute now
              val = 0;
#pragma unroll
              for(int k = 0; k < SMLS; k++) {
                  val += local_input[ix][k] * local_weights[iy][k]; // column-major
              }
              local_output[ix][iy] += val;
          }
      }

      // now push the local_output into the global memory now

      for( int itid = tid; itid < SMLS * SMLS; itid += blockDim.x) {
        // keep warp in row
        i_block_x = itid / SMLS;
        i_block_y = itid % SMLS;
        if (block_x * SMLS + i_block_x < batch && block_y * SMLS  + i_block_y < output_dim) {
          output[block_x * SMLS + i_block_x][block_y * SMLS + i_block_y] = local_output[i_block_x][i_block_y];
        }
      }
  }
}


torch::Tensor rz_linear_forward_cuda_tiled (
    const torch::Tensor& hashed_weights, // 1 x n
    const torch::Tensor& input, // b x d1
    const torch::Tensor& random_numbers,
    int input_dim,
    int output_dim,
    int chunk_size
    )
{
    int M = input.size(0);
    int K = input_dim;
    int N = output_dim;

    int64_t hashedWeightSize = hashed_weights.size(0);
    auto output = at::empty({input.size(0), output_dim}, input.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    dim3 block = dim3(SMLS, SMLS, 1);

    int total_number_of_blocks = (int)((M + SMLS - 1) / SMLS) * (int)((N + SMLS - 1) / SMLS) ;
    int grid = MAX_GRID_SIZE;
    if (total_number_of_blocks < MAX_GRID_SIZE) {
        grid = total_number_of_blocks;
    }

    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "rz_linear_forward_cuda", ([&] {
        rz_linear_forward_cuda_kernel_tiled<scalar_t><<<grid, block, 0, stream>>>(
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            M,
            K,
            N,
            chunk_size,
            hashed_weights.size(0)
      );
    }));
   return output;
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor rz_linear_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& input,
    const torch::Tensor& random_numbers,
    int input_dim,
    int output_dim,
    int chunk_size,
    bool tiled
)
{
  
    CHECK_INPUT(hashed_weights);
    CHECK_INPUT(input);
    CHECK_INPUT(random_numbers);
    if (tiled) {
        return rz_linear_forward_cuda_tiled(hashed_weights, input, random_numbers, input_dim, output_dim, chunk_size);
    } else {
        return rz_linear_forward_cuda(hashed_weights, input, random_numbers, input_dim, output_dim, chunk_size);
    }
}



std::tuple<torch::Tensor, torch::Tensor> rz_linear_backward_cuda (
    const torch::Tensor& out_grad,
    const torch::Tensor& hashed_weights, // 1 x n
    const torch::Tensor& input, // b x d1
    const torch::Tensor& random_numbers,
    int input_dim,
    int output_dim,
    int chunk_size,
    bool tiled
    )
{
    // we have to return two grad - w.r.t input and w.r.t hashed_weights

    auto input_grad = at::zeros({input.size(0), input.size(1)}, input.options());
    auto weight_grad = at::zeros({hashed_weights.size(0)}, input.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.device().index());

    //input_grad TODO cannot take advantage of chunk in robe-z here.
    int x_max = input.size(0);
    int y_max = input_dim;

    dim3 block = dim3(1, MAX_BLOCK_SIZE, 1);
    if (y_max < MAX_BLOCK_SIZE) {
        block = dim3(1, y_max, 1);
    }
    dim3 grid = dim3(MAX_GRID_SIZE, 1, 1);
    if ( x_max < MAX_GRID_SIZE) {
        grid = dim3(x_max, 1, 1);
    }
    
    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "rz_linear_backward_cuda_input", ([&] {
        rz_linear_backward_cuda_kernel_input<scalar_t><<<grid, block, 0, stream>>>(
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            out_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            input_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            input.size(0),
            input_dim,
            output_dim,
            chunk_size,
            hashed_weights.size(0),
            tiled
      );
    }));

    //weight_grad TODO cannot take advantage of chunk in robe-z here.
    x_max = input_dim;
    y_max = output_dim;

    block = dim3(1, MAX_BLOCK_SIZE, 1);
    if (y_max < MAX_BLOCK_SIZE) {
        block = dim3(1, y_max, 1);
    }
    grid = dim3(MAX_GRID_SIZE, 1, 1);
    if ( x_max < MAX_GRID_SIZE) {
        grid = dim3(x_max, 1, 1);
    }

    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "rz_linear_backward_cuda_weight", ([&] {
        rz_linear_backward_cuda_kernel_weight<scalar_t><<<grid, block, 0, stream>>>(
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            out_grad.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weight_grad.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            input.size(0),
            input_dim,
            output_dim,
            chunk_size,
            hashed_weights.size(0),
            tiled
      );
    }));
   return std::tuple<torch::Tensor, torch::Tensor>(input_grad, weight_grad);
}



std::tuple<torch::Tensor, torch::Tensor> rz_linear_backward(
    const torch::Tensor& out_grad,
    const torch::Tensor& hashed_weights,
    const torch::Tensor& input,
    const torch::Tensor& random_numbers,
    int input_dim,
    int output_dim,
    int chunk_size,
    bool tiled
)
{
  
    CHECK_INPUT(hashed_weights);
    CHECK_INPUT(input);
    CHECK_INPUT(out_grad);
    CHECK_INPUT(random_numbers);
    return rz_linear_backward_cuda(out_grad, hashed_weights, input, random_numbers, input_dim, output_dim, chunk_size, tiled);
}



__global__ void rz_linear_idx(torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits>  random_numbers,
                              torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits>  IDX,
                              int input_dim,
                              int output_dim,
                              int chunk_size,
                              int weight_size,
                              bool tiled) {
    int ty = threadIdx.x; // will be used for y dim initialized as x
    int bx = blockIdx.x;
    int64_t loc;
    for(; ty < output_dim; ty += blockDim.x) {
        for (; bx < input_dim; bx += gridDim.x) {
            if (tiled) {
                loc = location_tiled(bx, ty, random_numbers, weight_size);
            } else {
                loc = location(bx, ty, chunk_size, random_numbers, weight_size);
            }
            IDX[bx][ty] =  loc;
        }
    } 
}

torch::Tensor rz_get_idx(torch::Tensor& random_numbers, int input_dim, int output_dim, int chunk_size, int weight_size,  bool tiled) {
    CHECK_INPUT(random_numbers);

    auto IDX = at::zeros({input_dim, output_dim}, random_numbers.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(random_numbers.device().index());
    int block = MAX_BLOCK_SIZE;
    int grid = MAX_GRID_SIZE;
    if (block > output_dim) {
        block = output_dim;
    }
    if (grid > input_dim) {
        grid = input_dim;
    }

    rz_linear_idx<<<grid, block, 0, stream>>>(
        random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
        IDX.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
        input_dim,
        output_dim,
        chunk_size,
        weight_size,
        tiled
        );
    return IDX;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &rz_linear_forward, "robe_z_mm (CUDA)");
  m.def("backward", &rz_linear_backward, "robe_z_mm (CUDA)");
  m.def("get_idx", &rz_get_idx, "robe_z_mm (CUDA) ");
}
