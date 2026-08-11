# 0927_184211 gen by commit: 38655c
import os

LOCAL = True if os.environ.get("ZZ", "") else False
os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"
import torch
import dataclasses
from task import input_t, output_t
import time
from torch.utils.cpp_extension import load_inline
import sys
import threading
import torch.distributed as dist
from pathlib import Path
# import torch.distributed as dist


@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.float16
    out_dtype: torch.dtype = torch.float16


try:
    from utils import log
except Exception:
    log = print


CPP_WRAPPER = r"""// #define ZZ_DEBUG
#include <ATen/core/TensorBase.h>
#include <Python.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Half.h>
#include <cuda_fp16.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/library.h>
extern "C" void solve(const half *dp_x, int32_t *d_indices, half *d_ret,
                      half *d_debug_ret, float *d_expert_weights,
                      int num_tokens, int hidden_dim, int expert_per_token,
                      int num_local_experts, int local_rank, int32_t *raw);

extern "C" void init_shmem();
extern "C" void clear_shmem();

std::tuple<at::Tensor, at::Tensor>
tensor_base_add(const at::Tensor &dp_x, at::Tensor &indices,
                at::Tensor &expert_weights,
                /* TODO: this can be passed optimized, 提前传进来*/
                int64_t num_local_experts, int64_t local_rank) {

  // Validate inputs

  // Set CUDA device and launch kernel
  const at::cuda::CUDAGuard device_guard(dp_x.device());
  CHECK(dp_x.is_contiguous());
  CHECK(indices.is_contiguous());
  // dp_x.sizes(): [num_tokens, hidden_dim]
  int64_t num_tokens = dp_x.sizes()[0];
  int64_t hidden_dim = dp_x.sizes()[1];
  int64_t expert_per_token = indices.sizes()[1];
  at::Tensor ret = torch::empty({num_tokens, hidden_dim}, dp_x.options());

  at::Tensor debug_ret = at::Tensor();
  //     torch::empty({num_tokens, expert_per_token, hidden_dim},
  //     dp_x.options());

  at::Tensor raw = torch::empty({8 * 2}, indices.options());
  solve(reinterpret_cast<half *>(dp_x.data_ptr<c10::Half>()),
        indices.data_ptr<int32_t>(),
        reinterpret_cast<half *>(ret.data_ptr<c10::Half>()), nullptr,
        // reinterpret_cast<half *>(debug_ret.data_ptr<c10::Half>()),
        expert_weights.data_ptr<float>(), num_tokens, hidden_dim,
        expert_per_token, num_local_experts, local_rank,
        raw.data_ptr<int32_t>()); // 传参约定一下...

  return {ret, debug_ret};
}

// static bool sheMemInitd = false;
void clear_shmem_wrapper() { clear_shmem(); }
void init_shmem_wrapper() { init_shmem(); }
TORCH_LIBRARY(my_ops, m) {
  m.def("tensor_base_add", &tensor_base_add);
  m.def("clear_shmem", &clear_shmem_wrapper);
  m.def("init_shmem", &init_shmem);
}
PyMODINIT_FUNC PyInit_noname(void) {
  static struct PyModuleDef foo = {PyModuleDef_HEAD_INIT, "no_name", nullptr,
                                   -1, nullptr};
  return PyModule_Create(&foo);
}
"""

CUDA_SRC = r"""// #define ZZ_DEBUG
// #define USE_PERF 1
//

#pragma once
#define DEBUG_COMBINE 0
#define DEBUG_DISPATCH_READY 0
#define DEBUG_DISPATCH_READY2 0
#define DEBUG_COMBINE_READY 0
#define DEBUG_RAW_RET 0
#define CALL_PRINT 0
#define DEBUG_VALUE 0
#define DEBUG_LOCAL_DISPATCH_START_END 0

#ifdef __HIPCC__
// AMD ROCm HIP 平台
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define uni(func) hip##func
#define WARP_SYNC __builtin_amdgcn_wave_barrier();
#elif defined(__CUDACC__)
// NVIDIA CUDA 平台
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define uni(func) cuda##func
#define WARP_SYNC __syncwarp();
#else
#error "Unknown GPU platform"
#endif

#define _(a, b, c) (a) * (c) + (b)
#define DIV_UP(x, y) (x + y - 1) / y
constexpr int BLOCKSIZE = 128;

#define DEBUG_PRINT 1
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <unistd.h>
#ifndef ZZ
#define ZZ
#endif

static int MYRANK = -1;

#define UNI_CHECK(err)                                                         \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << std::endl;                     \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define UNI_CHECK_RANK(err, local_rank)                                        \
  {                                                                            \
    uni(Error_t) err_ = (err);                                                 \
    if (err_ != uni(Success)) {                                                \
      std::cerr << "UNI Error at " << __FILE__ << ":" << __LINE__ << " - "     \
                << uni(GetErrorString)(err_) << " rank: " << local_rank        \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
const char *IPC_HANDLE_FILENAME_RANK[] = {
    "ipc_handles_rank0.bin", "ipc_handles_rank1.bin", "ipc_handles_rank2.bin",
    "ipc_handles_rank3.bin", "ipc_handles_rank4.bin", "ipc_handles_rank5.bin",
    "ipc_handles_rank6.bin", "ipc_handles_rank7.bin"};

enum class FLAG {
  WAITING = 0,
  READY = 1,
};
enum class DATA_FLAG {
  WAITING = 0,
  READY = 1,
};

// void my_barrier_from_cpp() {
//   // 获取 Python 已经初始化好的默认进程组
//   UNI_CHECK(uni(DeviceSynchronize)());
//   auto pg = c10d::currentProcessGroup();
//   if (!pg) {
//       throw std::runtime_error("Default ProcessGroup is not initialized!");
//   }
//   // 调用 barrier 并等待完成
//   pg->barrier()->wait();
// }
constexpr int HIDDEN_DIM = 7168;
constexpr int MAX_TOKEN_NUM = 256;
constexpr int MAX_EXPERT_PER_TOKEN = 8;
constexpr int DISPATCH_CTA_THREADS = 1024;
constexpr int DISPATCH_THREAD_PER_EP = 8; // tmp TODO(change to 32) will increase the latency ???? why?
// PER_EP: 64 ⏱ 1475 ± 22.2 µs  slower then 8 & 32, so there are some sync issues here. try fix it first.
constexpr int DISPATCH_EP_NUM = DISPATCH_CTA_THREADS / DISPATCH_THREAD_PER_EP;
constexpr int RANK_SIZE = 8;
constexpr int DISPATCH_EP_BLOCK =
    MAX_TOKEN_NUM * MAX_EXPERT_PER_TOKEN / DISPATCH_EP_NUM;

static_assert(DISPATCH_THREAD_PER_EP <= 64,
              "DISPATCH_THREAD_PER_EP must be less than 32 for the syncwarp, "
              "maybe 64 for amd?");
static_assert(DISPATCH_THREAD_PER_EP * DISPATCH_EP_NUM <= 1024,
              "TOTAL NUM should small than 1024");
#define PERF_RANK (MYRANK == 3 && hidden_dim == HIDDEN_DIM)

struct DataBlock {
  float4 tensor_data[HIDDEN_DIM / 8];
  DATA_FLAG full_write_flag;
  int64_t pos;
};
struct Remote {
  int rank_token_num;
  FLAG rank_token_num_flag;
  DataBlock data_blocks[MAX_TOKEN_NUM * 4 /
                        DISPATCH_THREAD_PER_EP]; // remote_rank_idx here.
                                                 // why 4 here????
};

struct Local {
  DataBlock data_blocks[MAX_TOKEN_NUM * MAX_EXPERT_PER_TOKEN];
};
struct IpcCommBlock {
  int barrier[10000][64];
  Remote remote_dispatch_addr__rank_kid
      [RANK_SIZE][DISPATCH_EP_BLOCK]; // 8 ranks, each have 2 dispatch
                                      // kernel for atomic add.
  Local local_combine_addr;
};

// maybe we need a double buffer here later?? for the memset to execute in the
// background.

IpcCommBlock
    *h_remote_data[RANK_SIZE][RANK_SIZE]; // Host端指针数组，固定大小避免动态分配问题
static IpcCommBlock **d_remote_data[8]; // Device端指针，指向device内存中的指针数组

// Create separate streams for overlapping computation and communication
uni(Stream_t) stream_remote_combine, stream_local_combine, stream_default;

extern "C" void clear_shmem() {
  // // fprintf(stderr, "start clear_shmem\n");
  // UNI_CHECK(uni(Memset)(d_local_data, 0, sizeof(IpcCommBlock)));
  // UNI_CHECK(uni(DeviceSynchronize)());
  // // fprintf(stderr, "end clear_shmem\n");
}

__device__ static void device_sleep(int i) {
  return;
#ifdef ZZ_DEBUG
  if (threadIdx.x == 0)
    printf("start sleep\n");
  __syncthreads();
  unsigned long long start_clock = clock64();
  unsigned long long wait_clocks = (unsigned long long)(2 * 1.3e9);
  // 1.3e9 假设 GPU 时钟约 1.3 GHz，实际需用 uni(DeviceProp) 里的 clockRate 调整

  while (clock64() - start_clock < wait_clocks) {
    // busy-wait
  }
  __syncthreads();
  if (threadIdx.x == 0)
    printf("end sleep\n");
#endif
}
#ifdef __HIPCC__
template <typename U> __device__ static U load_volatile(U *src) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
  if (sizeof(U) == 1)
    u1 = __atomic_load_n((uint8_t *)src, __ATOMIC_ACQUIRE);
  else if (sizeof(U) == 2)

    u2 = __atomic_load_n((uint16_t *)src, __ATOMIC_ACQUIRE);
  else if (sizeof(U) == 4)
    u4 = __atomic_load_n((uint32_t *)src, __ATOMIC_ACQUIRE);
  else
    u8 = __atomic_load_n((uint64_t *)src, __ATOMIC_ACQUIRE);
#endif
  return elt;
}

template <typename U> __device__ static void store_volatile(U *dst, U val) {
  union {
    U elt;
    uint8_t u1;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  if (sizeof(U) == 1)
    __atomic_store_n((uint8_t *)dst, u1, __ATOMIC_RELEASE);
  else if (sizeof(U) == 2)
    __atomic_store_n((uint16_t *)dst, u2, __ATOMIC_RELEASE);
  else if (sizeof(U) == 4)
    __atomic_store_n((uint32_t *)dst, u4, __ATOMIC_RELEASE);
  else
    __atomic_store_n((uint64_t *)dst, u8, __ATOMIC_RELEASE);
}
#else
template <typename U> __device__ static U load_volatile(U *src) {
  union {
    U elt;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  if (sizeof(U) == 1)
    asm volatile("ld.volatile.global.b8 %0,[%1];"
                 : "=r"(u4)
                 : "l"(src)
                 : "memory");
  else if (sizeof(U) == 2)
    asm volatile("ld.volatile.global.b16 %0,[%1];"
                 : "=h"(u2)
                 : "l"(src)
                 : "memory");
  else if (sizeof(U) == 4)
    asm volatile("ld.volatile.global.b32 %0,[%1];"
                 : "=r"(u4)
                 : "l"(src)
                 : "memory");
  else
    asm volatile("ld.volatile.global.b64 %0,[%1];"
                 : "=l"(u8)
                 : "l"(src)
                 : "memory");
  return elt;
}

template <typename U> __device__ static void store_volatile(U *dst, U val) {
  union {
    U elt;
    uint16_t u2;
    uint32_t u4;
    uint64_t u8;
  };
  elt = val;
  if (sizeof(U) == 1)
    asm volatile("st.volatile.global.b8 [%0],%1;" ::"l"(dst), "r"(u4)
                 : "memory");
  else if (sizeof(U) == 2)
    asm volatile("st.volatile.global.b16 [%0],%1;" ::"l"(dst), "h"(u2)
                 : "memory");
  else if (sizeof(U) == 4)
    asm volatile("st.volatile.global.b32 [%0],%1;" ::"l"(dst), "r"(u4)
                 : "memory");
  else
    asm volatile("st.volatile.global.b64 [%0],%1;" ::"l"(dst), "l"(u8)
                 : "memory");
}
#endif

__global__ void ipc_handshake_kernel(IpcCommBlock **d_remote_data, int my_rank,
                                     int num_ranks) {
  // 每个线程检查一个peer
  int peer_rank = threadIdx.x;
  if (peer_rank >= num_ranks || peer_rank == my_rank) {
    return;
  }

  // 尝试读取远端内存的一个已知位置，比如barrier数组的第一个整数
  // 这里我们只是读取，不写入，目的是触发一次P2P访问
  // load_volatile可以确保这个读取不会被编译器优化掉
  int remote_value = load_volatile(&d_remote_data[peer_rank]->barrier[9000][0]);
}

extern "C" void init_shmem() {
  int my_rank = atoi(getenv("RANK"));
  MYRANK = my_rank;
  // fprintf(stderr, "my_rank: %d\n", my_rank);

  // 初始化主机端指针数组

  { // open other ranks' ipc handle file
    for (int kk = 0; kk < 8; kk++) {
      UNI_CHECK(uni(SetDevice)(kk));
      for (int rank = 0; rank < 8; rank++) {
        auto ipc_file_name = IPC_HANDLE_FILENAME_RANK[rank];
        int cnt = 0;
        while (access(ipc_file_name, F_OK) == -1) {
          usleep(100000);
          if (++cnt % 1000 == 0)
            fprintf(stderr, "waiting for ipc handle file %s\n", ipc_file_name);
        }
        uni(IpcMemHandle_t) ipc_handle;
        std::ifstream handle_file(ipc_file_name, std::ios::binary);
        handle_file.read(reinterpret_cast<char *>(&ipc_handle),
                         sizeof(ipc_handle));
        handle_file.close();

        // 修复IPC句柄的使用方式
        UNI_CHECK_RANK(uni(IpcOpenMemHandle)(
                           reinterpret_cast<void **>(&h_remote_data[kk][rank]),
                           ipc_handle, uni(IpcMemLazyEnablePeerAccess)),
                       my_rank);
      }
      UNI_CHECK(uni(Malloc)((void **)&d_remote_data[kk],
                            sizeof(IpcCommBlock *) * 8));
      UNI_CHECK(uni(Memcpy)(d_remote_data[kk], h_remote_data[kk],
                            sizeof(IpcCommBlock *) * 8,
                            uni(MemcpyHostToDevice)));
    }
  }
  { 
    UNI_CHECK(uni(SetDevice)(my_rank)); // 分别拷贝到每台机器上
    // 验证复制是否成功
    // sleep(8);
    fprintf(stderr, "my_rank: %d, Copied %d pointers to device memory\n",
            my_rank, 8);

    int my_rank = atoi(getenv("RANK"));
    int num_gpus = 0;
    UNI_CHECK(uni(GetDeviceCount)(&num_gpus));

    // 确保当前rank的device是正确的
    // UNI_CHECK(uni(SetDevice)(my_rank));

    fprintf(stderr,
            "Rank %d: Explicitly enabling peer access to all other ranks...\n",
            my_rank);
            

    // 等待所有设备操作完成
    if(0){
      ipc_handshake_kernel<<<1, 8>>>(d_remote_data[my_rank], my_rank, 8);
      UNI_CHECK(uni(DeviceSynchronize)());
      fprintf(stderr, "Rank %d: Peer access enablement finished.\n", my_rank);
    }
  }
  {

    // TODO: add a small kernel to access each other, to disable the
    // uni(IpcMemLazyEnablePeerAccess)
  }
  UNI_CHECK(uni(StreamCreate)(&stream_remote_combine));
  UNI_CHECK(uni(StreamCreate)(&stream_local_combine));
  UNI_CHECK(uni(StreamCreate)(&stream_default));
}

__global__ void local_dispatch(IpcCommBlock **d_remote_data, const half *dp_x,
                               const int32_t *d_indices, const int num_tokens,
                               const int hidden_dim, const int expert_per_token,
                               const int num_local_experts,
                               const int local_rank) {
  // 每 DISPATCH_THREAD_PER_EP 线程处理一个 expert_idx,
  // 每个 block 处理 DISPATCH_EP_NUM 个 expert_idx,
  // 所以一共启动 DISPATCH_EP_BLOCK 个 CTA.

  // 因此接收方呢? 相同的方案, 但是要启动 8 倍对吧
#if DEBUG_LOCAL_DISPATCH_START_END == 1
  if (threadIdx.x == 0) {
    printf("call local dispatch\n");
  }
#endif
#define IS_EXPERT_FIRST_IDX (threadIdx.x % DISPATCH_THREAD_PER_EP == 0)

  const int global_expert_idx =
      blockIdx.x * (DISPATCH_EP_NUM) + threadIdx.x / DISPATCH_THREAD_PER_EP;

  __shared__ int rank_token_num[RANK_SIZE];
  // __shared__ int rank_token_offset[8][1024]; // TODO 后续优化, 后面再实现,
  if (threadIdx.x < RANK_SIZE) // RANK_NUM
    rank_token_num[threadIdx.x] = 0;
  __syncthreads();
  // can optimize one register use here using template.
  int dst_rank = 0;
  if (global_expert_idx < num_tokens * expert_per_token){
    dst_rank = d_indices[global_expert_idx] / num_local_experts;
  }
  __shared__ int shared_expert_idx[DISPATCH_EP_NUM];
  if ((global_expert_idx < num_tokens * expert_per_token) &&
      IS_EXPERT_FIRST_IDX) {
    const int remote_token_idx = atomicAdd(&rank_token_num[dst_rank], 1);
    shared_expert_idx[threadIdx.x / DISPATCH_THREAD_PER_EP] = remote_token_idx;
#if DEBUG_COMBINE == 1
    printf("bid:%d, tid:%d, write to %d value: %d\n", blockIdx.x, threadIdx.x,
           threadIdx.x / DISPATCH_THREAD_PER_EP,
           shared_expert_idx[threadIdx.x / DISPATCH_THREAD_PER_EP]);
#endif
  }
  __syncthreads(); // TODO:__syncwarp();
  if (threadIdx.x < 8) {
    store_volatile(&d_remote_data[threadIdx.x]
                        ->remote_dispatch_addr__rank_kid[local_rank][blockIdx.x]
                        .rank_token_num,
                   rank_token_num[threadIdx.x]);
#ifdef ZZ_DEBUG
    printf("set remote rank %d, pos: %d to token_num: %d\n", threadIdx.x,
           blockIdx.x, rank_token_num[threadIdx.x]);
#endif
    __threadfence_system();
    store_volatile(&d_remote_data[threadIdx.x]
                        ->remote_dispatch_addr__rank_kid[local_rank][blockIdx.x]
                        .rank_token_num_flag,
                   FLAG::READY);
#if DEBUG_DISPATCH_READY == 1
    printf("set:: DISPATCH_BLOCK:: remote rank: %d, local rank: %d, block: %d "
           "to READY\n",
           threadIdx.x, local_rank, blockIdx.x);
#endif
  }
  WARP_SYNC;
  if (global_expert_idx < num_tokens * expert_per_token) {
    const int remote_token_idx =
        shared_expert_idx[threadIdx.x / DISPATCH_THREAD_PER_EP];
#ifdef DEBUG_COMBINE
    // printf("bid:%d, tid:%d, read from %d value: %d\n", blockIdx.x,
    // threadIdx.x,
    //        threadIdx.x / DISPATCH_THREAD_PER_EP, remote_token_idx);
#endif
    const int token_idx = global_expert_idx / expert_per_token;
    const float4 *local_pos = (float4 *)&(dp_x[token_idx * hidden_dim]);
#ifdef DEBUG_COMBINE
    // printf("write remote rank %d, local_rank: %d, blockIdx.x: %d , "
    //        "remote_token_idx: %d from token_idx: %d\n",
    //        dst_rank, local_rank, blockIdx.x, remote_token_idx, token_idx);
#endif
    for (int i = threadIdx.x % DISPATCH_THREAD_PER_EP; i < hidden_dim / 8;
         i += DISPATCH_THREAD_PER_EP) { // /8 here for fp16
      const float4 data = local_pos[i];
      d_remote_data[dst_rank]
          ->remote_dispatch_addr__rank_kid[local_rank][blockIdx.x]
          .data_blocks[remote_token_idx]
          .tensor_data[i] = data; // here have a bug, a bad address access.
      WARP_SYNC;
    }
    if (IS_EXPERT_FIRST_IDX) {
      d_remote_data[dst_rank]
          ->remote_dispatch_addr__rank_kid[local_rank][blockIdx.x]
          .data_blocks[remote_token_idx]
          .pos = global_expert_idx;
    }

    // if (dst_rank > 8) { // TODO: refactor to check for this?????????????? and
    //                     // how to abort...
    //   for (int i = 0; i < 1000; i++) {
    //     printf("should not happen here..");
    //   }
    // }
    __threadfence_system();
    WARP_SYNC;
    if (IS_EXPERT_FIRST_IDX) {
      store_volatile(
          &d_remote_data[dst_rank]
               ->remote_dispatch_addr__rank_kid[local_rank][blockIdx.x]
               .data_blocks[remote_token_idx]
               .full_write_flag,
          DATA_FLAG::READY);
#if DEBUG_DISPATCH_READY2 == 1
      printf("write dst_rank: %d, local_rank: %d, blockIdx.x: %d, "
             "remote_token_idx: %d to full_write_flag\n",
             dst_rank, local_rank, blockIdx.x, remote_token_idx);
#endif
    }
  }
#if DEBUG_LOCAL_DISPATCH_START_END == 1
  if (threadIdx.x == 0) {
    printf("call local dispatch end\n");
  }
#endif
  __threadfence_system();
}

__global__ void remote_combine(IpcCommBlock **d_remote_data,
                               const int local_rank, const int hidden_dim) {
  const int bid = blockIdx.x;
#if ZZ_DEBUG == 1
  if (threadIdx.x == 0)
    printf("start remote_combine: %d\n", bid);
#endif
  const int rank_to_process = bid / DISPATCH_EP_BLOCK;
#define LOCAL_ADDR                                                             \
  d_remote_data[local_rank]                                                    \
      ->remote_dispatch_addr__rank_kid[rank_to_process]                        \
                                      [bid % DISPATCH_EP_BLOCK]
  int64_t cnt_dispatch = 0;
  if (threadIdx.x == 0) {
#if ZZ_DEBUG == 1
    printf("bid:%d wait for rank %d to be READY %ld\n", blockIdx.x,
           rank_to_process,
           (void **)&LOCAL_ADDR.rank_token_flag -
               (void **)d_remote_data[local_rank]);
    int64_t i = 0;
#endif
    auto &local_addr =
        d_remote_data[local_rank]
            ->remote_dispatch_addr__rank_kid[rank_to_process]
                                            [bid % DISPATCH_EP_BLOCK];
    while (true) {
      if (load_volatile(&LOCAL_ADDR.rank_token_num_flag) == FLAG::READY) {
        break;
      }
#if DEBUG_DISPATCH_READY == 1
      if (++cnt_dispatch % 10000000LL == 0) {
        printf("read:: DISPATCH_BLOCK:: remote rank: %d, local rank: %d, "
               "block: %d "
               "to READY\n",
               local_rank, rank_to_process, bid % DISPATCH_EP_BLOCK);
        // printf("remote_combine still wait local_rank: %d, rank_to_process: %d
        // :%d wait for rank %d "
        //        "to be READY %ld\n",
        //        LOCAL_ADDR.rank_token_flag, blockIdx.x, rank_to_process,
        //        (void **)&LOCAL_ADDR.rank_token_flag -
        //            (void **)d_remote_data[local_rank]);
      }
#endif
    }
    store_volatile(&LOCAL_ADDR.rank_token_num_flag, FLAG::WAITING);
  }
  __threadfence_system();

  __shared__ int rank_token_num[1];
  rank_token_num[0] =
      load_volatile(&LOCAL_ADDR.rank_token_num); // TODO: why need shared mem
                                                 // here? and all reads here?
  // LOCAL_ADDR.rank_token_num = 0;
  __syncthreads();
  auto local_expert_idx = threadIdx.x / DISPATCH_THREAD_PER_EP;
  if (local_expert_idx >= rank_token_num[0]) {
    return;
  }
  __threadfence_system();
  if (IS_EXPERT_FIRST_IDX) { // maybe all doing this is good?
    while (true) {
      if (load_volatile(
              &LOCAL_ADDR.data_blocks[local_expert_idx].full_write_flag) ==
          DATA_FLAG::READY) {
        break;
      }
#if DEBUG_DISPATCH_READY2 == 1
      if (++cnt_dispatch % 10000000LL == 0) {
        printf(
            "DISPATCH_EP_READ:: dst_rank: %d, local_rank: %d, blockIdx.x: %d, "
            "remote_token_idx: %d to full_write_flag with theradIdx.x %d\n",
            local_rank, rank_to_process, bid % DISPATCH_EP_BLOCK,
            local_expert_idx, threadIdx.x);
      }
#endif
    }
  }
  // __threadfence(); // this can be optimized with no degen
  // LOCAL_ADDR.data_blocks[tid].full_write_flag = DATA_FLAG::WAITING;
  // __syncthreads();
  WARP_SYNC; // TODO: should be syncwrap(); here if remove will be 40% slower...
             // so right, the thread
             //  divergence.
  if (IS_EXPERT_FIRST_IDX) {
    store_volatile(&LOCAL_ADDR.data_blocks[local_expert_idx].full_write_flag,
                   DATA_FLAG::WAITING);
  }
  WARP_SYNC; // TODO: should be syncwrap(); here if remove will be 40% slower...
  __threadfence_system();
  const int remote_pos =
      load_volatile(&LOCAL_ADDR.data_blocks[local_expert_idx].pos);
  const float factor = (1 + local_rank);
  half factor_half = __float2half(factor);
  half2 factor_half2 = __half2half2(factor_half);
  for (int i = threadIdx.x % DISPATCH_THREAD_PER_EP; i < hidden_dim / 8;
       i += DISPATCH_THREAD_PER_EP) {
    float4 data = LOCAL_ADDR.data_blocks[local_expert_idx].tensor_data[i];
    half2 *data_as_half2 = (half2 *)&data;

    data_as_half2[0] = __hmul2(data_as_half2[0], factor_half2);
    data_as_half2[1] = __hmul2(data_as_half2[1], factor_half2);
    data_as_half2[2] = __hmul2(data_as_half2[2], factor_half2);
    data_as_half2[3] = __hmul2(data_as_half2[3], factor_half2);

    LOCAL_ADDR.data_blocks[local_expert_idx].tensor_data[i] = data;
    d_remote_data[rank_to_process]
        ->local_combine_addr.data_blocks[remote_pos]
        .tensor_data[i] = data;
  }
  WARP_SYNC;
  __syncthreads();
  __threadfence_system();

#if DEBUG_VALUE == 1
  const half2 *remote_value =
      (half2 *)&(d_remote_data[rank_to_process]
                     ->local_combine_addr.data_blocks[remote_pos]
                     .tensor_data[0]
                     .x);
  auto value_print = __half2float(remote_value[0].x);
  const half2 *local_value =
      (half2 *)&(LOCAL_ADDR.data_blocks[local_expert_idx].tensor_data[0].x);
  auto value_local = __half2float(local_value[0].x);
  printf("write remote bid:%d tid:%d, rank:%d, pos:%d val:%lf, local: %lf to "
         "READY\n",
         blockIdx.x, threadIdx.x, rank_to_process, remote_pos, value_print,
         value_local);
  if (0) {
    if (threadIdx.x == 0)
      printf("end remote_combine: %d\n", bid);
  }
#endif
  __threadfence_system();
  __syncthreads();
  WARP_SYNC;
  __threadfence_system();
  if (IS_EXPERT_FIRST_IDX) {
    store_volatile(&d_remote_data[rank_to_process]
                        ->local_combine_addr.data_blocks[remote_pos]
                        .full_write_flag,
                   DATA_FLAG::READY);
#if DEBUG_COMBINE_READY == 1
    printf("COMBINE_EP:: write remote rank: %d, remote_pos: %04d to "
           "full_write_flag with "
           "blockIdx:%d,  threadIdx: %d\n",
           rank_to_process, remote_pos, blockIdx.x, threadIdx.x);
#endif
  }
  __syncthreads();
  WARP_SYNC;
  __threadfence_system();
#undef LOCAL_ADDR
}

__global__ void local_combine_first(IpcCommBlock **d_remote_data,
                                    const int local_rank, const int num_tokens,
                                    const int expert_per_token) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (global_idx >= num_tokens)
    return; // TODO: optimize the parallel here..
  DataBlock *local_data =
      &(d_remote_data[local_rank]
            ->local_combine_addr.data_blocks[global_idx * expert_per_token]);
#ifdef ZZ_DEBUG
  // half2 *ret_data = (half2 *)&(d_ret[global_idx * hidden_dim]);
  // float *expert_weights_data =
  //     &(d_expert_weights[global_idx *
  //                        expert_per_token]); // TODO: can be refactor to
  // d_expert_weights += global_idx *
  // expert_per_token...
#endif
  for (int i = 0; i < expert_per_token; i++) {
    int64_t j = 0;
    while (true) {
      if (load_volatile(&local_data[i].full_write_flag) == DATA_FLAG::READY) {
        break;
        // maybe add a small usleep here??
        // and also, the memory is not collese here... a bad practice
        // but I don't want false sharing here... so more optimize.
      }
#if DEBUG_COMBINE_READY == 1
      if (j++ % 10000000LL == 0) {
        printf("COMBINE_EP:: read remote rank: %d, remote_pos: %04d to "
               "full_write_flag with "
               "blockIdx:%d,  threadIdx: %d\n",
               local_rank, global_idx * expert_per_token + i, blockIdx.x,
               threadIdx.x);
      }
#endif
    }
    WARP_SYNC;
    __threadfence_system();
    store_volatile(&local_data[i].full_write_flag, DATA_FLAG::WAITING);
    // local_data[i].full_write_flag = DATA_FLAG::WAITING;
  }
  __syncthreads();
  WARP_SYNC;
  __threadfence_system();
  // printf("local combine finished: %d\n", global_idx);
}

template <int TN = 256, int THREADS_PER_ELE = 32>
__global__ void local_combine(IpcCommBlock **d_remote_data, half *d_ret,
                              float const *d_expert_weights, half2 *d_debug_ret,
                              const int local_rank, const int hidden_dim,
                              const int num_tokens,
                              const int expert_per_token) {
  constexpr int BN = TN / THREADS_PER_ELE;
  static_assert(TN % BN == 0, "TN must be divisible by BN");

  // const int global_idx = blockIdx.x * TN + threadIdx.x;
  const int token_idx = (blockIdx.x * BN) + (threadIdx.x / THREADS_PER_ELE);
  const int thread_idx = threadIdx.x % THREADS_PER_ELE;
  if (token_idx >= num_tokens)
    return; // TODO: optimize the parallel here..
  DataBlock *local_data =
      &(d_remote_data[local_rank]
            ->local_combine_addr.data_blocks[token_idx * expert_per_token]);
  half2 *ret_data = (half2 *)&(d_ret[token_idx * hidden_dim]);
  d_expert_weights += token_idx * expert_per_token;
  float shared_expert_weights[8]; // is register enough?
  for (int i = 0; i < expert_per_token; i++) {
    shared_expert_weights[i] = d_expert_weights[i]; // TODO: float4?
  }
#if DEBUG_RAW_RET == 1
  d_debug_ret += token_idx * expert_per_token * (hidden_dim / 2);
#endif

  for (int i = thread_idx; i < hidden_dim / 2; i += THREADS_PER_ELE) {
    float x = 0, y = 0;
    for (int j = 0; j < expert_per_token; j++) {
      half2 data = ((half2 *)&local_data[j].tensor_data)[i];
#if DEBUG_RAW_RET == 1
      d_debug_ret[j * hidden_dim / 2 + i] = data;
#endif
      x += __half2float(data.x) * shared_expert_weights[j];
      y += __half2float(data.y) * shared_expert_weights[j];
    }
    ret_data[i] = make_half2(__float2half_rn(x), __float2half_rn(y));
  }
}
__global__ void dist_barrier(IpcCommBlock **d_remote_data, int index,
                             int local_rank) {
  int dst_rank = threadIdx.x / 64;
  if (threadIdx.x % 64 != 0) {
    return;
  }
  store_volatile(&d_remote_data[dst_rank]->barrier[index][local_rank], 1);
  __threadfence_system();
  while (true) {
    if (load_volatile(&d_remote_data[local_rank]->barrier[index][dst_rank]) ==
        1) {
      break;
    }
  }
}

// __global__ void dist_barrier(IpcCommBlock **d_remote_data, int index, int
// my_rank) {
//   // 只让一个线程执行 barrier 逻辑
//   if(threadIdx.x != 0) return;
//   printf("dist_barrier called with index: %d, my_rank: %d\n", index,
//   my_rank);

//   // 标记自己已到达所有其他 rank 的 barrier
//   for(int i = 0; i < 8; i++) {
//       if(i != my_rank) {
//           store_volatile(&d_remote_data[i]->barrier[index][my_rank], 1);
//       }
//   }
//   __threadfence_system();

//   // 等待所有其他 rank 都标记自己已到达
//   for(int i = 0; i < 8; i++) {
//       if(i != my_rank) {
//           while(load_volatile(&d_remote_data[my_rank]->barrier[index][i]) !=
//           1) {
//               // 等待
//           }
//       }
//   }
// }

static int barrier_index = 0;

extern "C" void solve(const half *dp_x, int32_t *d_indices, half *d_ret,
                      half *d_debug_ret, float *d_expert_weights,
                      int num_tokens, int hidden_dim, int expert_per_token,
                      int num_local_experts, int local_rank, int32_t *raw) {

  UNI_CHECK(uni(SetDevice)(local_rank));
  // UNI_CHECK(uni(MemcpyAsync)(raw, &h_remote_data[local_rank], 8 * 2 * sizeof(int32_t),
                  //  uni(MemcpyHostToDevice)));
  IpcCommBlock **d_remote_data_copy = d_remote_data[local_rank];
#if CALL_PRINT == 1
  // dp_x: [num_tokens, hidden_dim]
  // indices: [num_tokens, expert_per_token]
  // UNI_CHECK(uni(DeviceSynchronize)());
  // UNI_CHECK_RANK(uni(DeviceSynchronize)(), local_rank);
  // fprintf(stderr,
  //         "my_rank: %d, slove called with num_tokens: %d, hidden_dim: %d, "
  //         "expert_per_token: "
  //         "%d, num_local_experts: %d, local_rank: %d, \n",
  //         local_rank, num_tokens, hidden_dim, expert_per_token,
  //         num_local_experts, local_rank);
  // fprintf(stderr,
  //         "my_rank: %d, dp_x: %p, d_indices:%p, d_ret: %p, d_expert_weights: "
  //         "%p, d_remote_data: %p\n",
  //         local_rank, dp_x, d_indices, d_ret, d_expert_weights,
  //         d_remote_data_copy);
  // UNI_CHECK(uni(DeviceSynchronize)());
#endif

  local_dispatch<<<DISPATCH_EP_BLOCK,
                   DISPATCH_EP_NUM * DISPATCH_THREAD_PER_EP>>>(
      d_remote_data_copy, dp_x, d_indices, num_tokens, hidden_dim,
      expert_per_token, num_local_experts, local_rank);
#if CALL_PRINT == 1
  // UNI_CHECK_RANK(uni(DeviceSynchronize)(), local_rank);
  // fprintf(stderr, "my_rank: %d, local_dispatch done\n", local_rank);

#endif
  // usleep(100);
  dist_barrier<<<1, 64 * 8>>>(d_remote_data_copy, ++barrier_index, local_rank);
#if CALL_PRINT == 1
  UNI_CHECK(uni(DeviceSynchronize)());
  // fprintf(stderr, "my_rank: %d, dist_barrier done\n", local_rank);

#endif

  UNI_CHECK(uni(GetLastError)());
// #define STREAM stream_default
#if CALL_PRINT == 1
  // fprintf(stderr, "call remote_combine with grid: %d, block %d\n",
  //         RANK_SIZE * DISPATCH_EP_BLOCK,
  //         DISPATCH_EP_NUM * DISPATCH_THREAD_PER_EP);
#endif
  remote_combine<<<RANK_SIZE * DISPATCH_EP_BLOCK,
                   DISPATCH_EP_NUM * DISPATCH_THREAD_PER_EP>>>(
      d_remote_data_copy, local_rank, hidden_dim);
  UNI_CHECK(uni(GetLastError)());
  // sleep(0.1);
#if CALL_PRINT == 1
  // UNI_CHECK(uni(DeviceSynchronize)());
  // fprintf(stderr, "my_rank: %d, remote_combine done\n", local_rank);

#endif
  dist_barrier<<<1, 64 * 8>>>(d_remote_data_copy, ++barrier_index, local_rank);

  // 暂时展开成两阶段,方便进行 debug....
  // 然后后面可以试下把 local_combine 放到 main_stream里面,
  // 这样可以省去sync的操作, 后置一下
  // 这个 kernel 不影响, 因为不用分块..
  local_combine_first<<<DIV_UP(num_tokens, 256), 256>>>(
      d_remote_data_copy, local_rank, num_tokens, expert_per_token);
#ifdef USE_PERF
  uni(Event_t) event_start, event_end;
  if (PERF_RANK) {
    uni(EventCreate)(&event_start);
    uni(EventCreate)(&event_end);
    uni(EventRecord)(event_start, stream_local_combine);
  }
#endif
  constexpr int TN = 256;
  constexpr int THREADS_PER_ELE = 32;
  // dist_barrier<<<1, 64 * 8>>>(d_remote_data, ++barrier_index, local_rank);
  local_combine<TN, THREADS_PER_ELE>
      <<<DIV_UP(num_tokens * THREADS_PER_ELE, TN), TN>>>(
          d_remote_data_copy, d_ret, d_expert_weights, (half2 *)d_debug_ret,
          local_rank, hidden_dim, num_tokens, expert_per_token);
#ifdef USE_PERF
  if (PERF_RANK) {
    uni(EventRecord)(event_end, stream_local_combine);
    uni(EventSynchronize)(event_end);
    float elapsed_time;
    uni(EventElapsedTime)(&elapsed_time, event_start, event_end);
    printf("local_combine time: %f ms\n", elapsed_time);
  }
#endif
  // UNI_CHECK_RANK(uni(DeviceSynchronize)(), local_rank);
}

/* TODO:
    store_volatile(&LOCAL_ADDR.rank_token_num_flag, FLAG::WAITING); //
   写回的操作或许可以省掉? 我们每次让他 index+1 就行........  最后多申请一些
   buffer 就好.............. optimize later...... 然后初始的时候多拷贝一点 addr
   进去.


*/
"""

CUDA_MAIN_SRC = r"""#include "ref10_first.hip"
#include <ctime>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h> // for sleep()

bool isFileOlderThan_posix(const char *filePath, double seconds_threshold) {
  struct stat fileStat;

  // 2. 调用 stat 获取文件信息
  if (stat(filePath, &fileStat) != 0) {
    // 如果 stat 返回非0值，表示出错（如文件不存在）
    perror("stat 错误");
    return false;
  }

  // 1. 获取当前时间
  time_t currentTime = time(nullptr);

  // 3. 计算时间差
  double diff_seconds = difftime(currentTime, fileStat.st_mtime);

  // 4. 比较差值
  if (diff_seconds > seconds_threshold) {
    std::cout << "file '" << filePath << "' is older than "
              << seconds_threshold << " s, real " << diff_seconds
              << "s)\n";
    return true;
  } else {
    std::cout << "file '" << filePath << "' modified in " << seconds_threshold
              << " s, real " << diff_seconds << "s\n";
    return false;
  }
}

int main() {
  for (int i = 0; i < 8; i++) {
    IpcCommBlock *d_local_data = nullptr;
    UNI_CHECK(uni(SetDevice)(i));
    UNI_CHECK(hipExtMallocWithFlags((void**)&d_local_data, sizeof(IpcCommBlock), hipDeviceMallocFinegrained));
    UNI_CHECK(uni(Memset)(d_local_data, 0, sizeof(IpcCommBlock)));
    uni(IpcMemHandle_t) ipc_handle;
    UNI_CHECK(uni(IpcGetMemHandle)(&ipc_handle, d_local_data));
    std::string s = std::string(IPC_HANDLE_FILENAME_RANK[i]);
    std::ofstream handle_file(s, std::ios::binary);
    handle_file.write(reinterpret_cast<char *>(&ipc_handle),
                      sizeof(ipc_handle));
    handle_file.close();
    fprintf(stderr, "write ipc handle to %s\n", s.c_str());
  }
  fprintf(stderr, "start to sleep now\n");
  const char *filename = "now.txt";
  for (int i = 1; i < 10; i++) {
    constexpr int sleep_time = 5;
    sleep(5);
    if (isFileOlderThan_posix(filename, 10.0)) {
      fprintf(stderr, "file is older than 20s, break\n");
      break;
    }
    fprintf(stderr, "%d s sleeped\n ", i * sleep_time);
  }
}
"""

import os

if not os.environ.get("ZZ"):
    os.environ["CXX"] = "clang++"

log("compile and load start")
tic = time.time()
extra_cuda_cflags = [
    "-O3",  # TODO: try O3 later.... amd O3 maybe unstable
    "-DZZ=1",
    # "-g",
    # "-G",
]
if not LOCAL:
    extra_cuda_cflags.append("--offload-arch=gfx942")
    extra_cuda_cflags.append("--std=c++20")
else:
    extra_cuda_cflags.extend(
        [
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-lineinfo",
        ]
    )

# import fcntl


def main_func():
    module = load_inline(
        name="noname",
        cpp_sources=[CPP_WRAPPER],
        cuda_sources=[CUDA_SRC],
        # functions=['fp8_mm'],
        verbose=True,
        no_implicit_headers=True,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    return module


if False:
    import fcntl

    IS_FIST_RANK_NAME = "/tmp/is_first_rank.lock"
    is_first_rank = False
    fd = open(IS_FIST_RANK_NAME, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        is_first_rank = True
    except Exception:
        is_first_rank = False

    if is_first_rank:
        for i in os.environ:
            print("ENV: ", i, "=", os.environ[i], flush=True)


if not LOCAL:
    main_func()
else:

    def compile_and_load(file_name=None):
        from torch.utils.cpp_extension import load

        if file_name is None:
            file_name = os.path.splitext(os.path.basename(__file__))[0]
        # Get the directory name and join with the file name
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_name = os.path.join(current_dir, file_name)

        log("load module")
        tic = time.time()
        extra_cuda_flags = [
            "-O2",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-DZZ=1",
            # "-fsanitize=address",
            # "-fsanitize=thread",
            "-g",
            # "-G",
        ]
        if not torch.version.hip:
            extra_cuda_flags.append("-lineinfo")
        lib = load(
            name="noname",
            sources=[
                f"{file_name}.hip" if torch.version.hip else f"{file_name}.cu",
                f"{file_name}.cpp",
            ],
            extra_cuda_cflags=extra_cuda_flags,
            extra_cflags=["-std=c++17"],
        )
        log(f"compile use{time.time() - tic} s")
        return lib

    log("compile and load start")
    lib = compile_and_load()
if False:
    import fcntl

    LOCK_FILE = "/tmp/mylockfile.lock"
    fd = open(LOCK_FILE, "w")
    i = 0
    while True:
        try:
            # 非阻塞模式获取独占锁
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # print(f"获取到锁了, 正常执行", time.time() - tic, flush=True)
            main_func()
            fcntl.flock(fd, fcntl.LOCK_UN)
            # print("正常执行结束, used time: ", time.time() - tic, flush=True)
            break
        except BlockingIOError:
            i += 1
            time.sleep(0.01)
            if i % 1000 == 0:
                print("waiting for lock...", time.time() - tic, i, flush=True)


def periodic_flush():
    while True:
        sys.stdout.flush()
        time.sleep(0.5)


threading.Thread(target=periodic_flush, daemon=True).start()


log("compile and load end use time: %f seconds" % (time.time() - tic))

now_cnt = 0


# class PyTorchAllToAll:
#     def __init__(self, cfg: MoEConfig, rank: int, world_size: int):
#         self.cfg = cfg
#         self.rank = rank
#         self.world_size = world_size
#         self.num_local_experts = cfg.num_experts // world_size
#         self.max_recv = cfg.max_num_tokens * world_size
#         self.send_meta = None  # recv 时候要用


def rank0_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank1_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank2_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank3_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank4_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank5_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank6_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def rank7_call(dp_x, indices, expert_weights, num_local_experts, rank):
    ret = torch.ops.my_ops.tensor_base_add(
        dp_x, indices, expert_weights, num_local_experts, rank
    )
    return ret


def one_kernel(
    dp_x: torch.Tensor,
    indices: torch.Tensor,
    expert_weights: torch.Tensor,
    num_local_experts: int,
    rank: int,
):
    # pg = torch.distributed.distributed_c10d._get_default_group()
    # torch._C._distributed_c10d._set_process_group(pg)
    # dist.barrier()
    if not hasattr(torch, "__meminit"):
        os.environ["RANK"] = str(rank)
        print(f"{time.time() - tic}s start init here. {rank=}", flush=True)
        if rank == 0:
            # 准备要执行的命令
            # 最好直接在 Python 中执行命令，而不是通过 shell 脚本
            # 但为了最小改动，我们仍然调用 run.sh
            os.system("rm -rf *.bin")
            time.sleep(1)
            if not LOCAL:
                with open("make_share.hip", "w") as f:
                    f.write(CUDA_MAIN_SRC)
                with open("ref10_first.hip", "w") as f:
                    f.write(CUDA_SRC)
                if not os.path.exists("a.out"):
                    os.system("hipcc make_share.hip && ./a.out &")
                else:
                    os.system("./a.out &")
            else:
                script_content = "nohup ./a.out&"
                os.system(script_content)

        try:
            dist.barrier()
            log(f"{time.time() - tic}s end barrier")
        except Exception:
            pass
        log(f"{time.time() - tic}s start init")
        torch.ops.my_ops.init_shmem()
        log(f"{time.time() - tic}s end init")
        torch.__meminit = True
        try:
            dist.barrier()
            log(f"{time.time() - tic}s end barrier")
        except Exception:
            pass
    else:
        # log(f"{time.time() - tic}s skip init shmem, already init")
        pass
    if rank == 0:
        p = Path("now.txt")
        p.touch()
        ret = rank0_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 1:
        ret = rank1_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 2:
        ret = rank2_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 3:
        ret = rank3_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 4:
        ret = rank4_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 5:
        ret = rank5_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 6:
        ret = rank6_call(dp_x, indices, expert_weights, num_local_experts, rank)
    elif rank == 7:
        ret = rank7_call(dp_x, indices, expert_weights, num_local_experts, rank)

    if not LOCAL:
        return ret[0]
    return ret[0]  ############## tmp


def custom_kernel(data: input_t) -> output_t:
    global tic
    cfg, rank_data, rank, world_size = data
    if rank == 0:
        # log(f"{time.time() - tic}s {rank=} {rank_data.x.shape=} start custom kernel")
        pass

    # ata = PyTorchAllToAll(cfg, rank, world_size)
    ret = one_kernel(
        rank_data.x,
        rank_data.indices,
        rank_data.weights,
        cfg.num_experts // world_size,
        rank,
    )
    # log(f"{time.time() - tic}s {rank=} end custom kernel")
    pass
    return ret


log(f"{time.time() - tic}s end load module")
