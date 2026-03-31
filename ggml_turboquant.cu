/*
 * TurboQuant: CUDA Kernel Implementation
 * ========================================
 * GPU-accelerated quantize/dequantize for KV cache vectors.
 *
 * Architecture notes for RTX 3090 (SM 8.6, Ampere):
 *   - 128 threads per block = 4 warps = good occupancy
 *   - Each thread handles one coordinate of the head_dim=128 vector
 *   - Rotation matrix loaded into shared memory (64 KB fits easily)
 *   - Codebook in constant memory (max 64 floats = 256 bytes)
 *
 * The key insight: TurboQuant's rotation step is a matrix-vector multiply
 * (Π · x), which maps naturally to one thread per output element with
 * shared-memory reduction. For d=128 this is one thread per coordinate.
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */
#include <stdio.h>
#include "ggml_turboquant.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

/* =========================================================================
 * Section 1: Constant Memory — Codebooks
 *
 * Codebooks are tiny (max 16 floats) and read-only, so constant memory
 * gives broadcast reads across all threads in a warp.
 * ========================================================================= */

__constant__ float d_codebook_3[8];
__constant__ float d_codebook_4[16];
// Precompute on host, pass as __constant__ memory.
// spread3[x] spreads 8 bits of x into 24 bits: bit k -> bit 3k
__constant__ unsigned int d_spread3[256];

/* =========================================================================
 * Section 2: Device Helper — Find Nearest Centroid
 * ========================================================================= */

__device__ __forceinline__
uint8_t tq_find_nearest(float val, const float * codebook, int n_levels) {
    float best_dist = 1e30f;
    uint8_t best_idx = 0;

    /* Codebook is sorted, so we could binary search.
     * But n_levels <= 16, so linear scan is faster due to no branching. */
    for (int c = 0; c < n_levels; c++) {
        float d = (val - codebook[c]);
        d = d * d;
        if (d < best_dist) {
            best_dist = d;
            best_idx = (uint8_t)c;
        }
    }
    return best_idx;
}

/* =========================================================================
 * Section 3: Quantize Kernel
 *
 * One block per vector (token×head). 128 threads per block (one per dim).
 *
 * Workflow:
 *   1. Load input vector into shared memory
 *   2. Compute L2 norm via warp reduction
 *   3. Each thread computes one element of y = Π · (x/||x||)
 *      by reading its row of Π from global memory
 *   4. Find nearest codebook centroid
 *   5. Bit-pack indices cooperatively
 *   6. Write output block
 * ========================================================================= */

__global__ void tq_quantize_kernel_tq3(
    const float * __restrict__ src,      /* [n_vectors × 128] input     */
    void        * __restrict__ dst,      /* [n_vectors × block_tq3] out */
    const float * __restrict__ rotation, /* [128 × 128] rotation matrix */
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;  /* 0..127, one per coordinate */
    const int d = TQ_HEAD_DIM;    /* 128 */

    unsigned int lane_id = tid % 32;
    int nwarps = blockDim.x / 32; /* Should be 4 for 128 threads */

    /* Shared memory: input vector + norm */
    __shared__ float s_input[TQ_HEAD_DIM];
    float s_norm_sq;

    /* Step 1: Load input vector */
    s_input[tid] = src[vec_idx * d + tid];
    __syncthreads();

    /* Step 2: Compute L2 norm via parallel reduction */
    float val_sq = s_input[tid] * s_input[tid];

    /* Warp-level reduction first */
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_sq += __shfl_down_sync(0xFFFFFFFF, val_sq, offset);
    }

    /* Cross-warp reduction: lane 0 of each warp writes to shared */
    __shared__ float s_warp_sums[4]; /* 128 threads / 32 = 4 warps */
    if (tid % 32 == 0) {
        s_warp_sums[tid / 32] = val_sq;
    }
    __syncthreads();

    s_norm_sq = s_warp_sums[0] + s_warp_sums[1] +
                s_warp_sums[2] + s_warp_sums[3];

    float norm = sqrtf(s_norm_sq);

    /* Handle zero vector */
    if (norm < 1e-15f) {
        if (tid == 0) {
            block_tq3 * blk = (block_tq3 *)((uint8_t *)dst +
                               vec_idx * sizeof(block_tq3));
            blk->norm = 0.0f;
            memset(blk->indices, 0, TQ3_INDEX_BYTES);
        }
        return;
    }

    float inv_norm = 1.0f / norm;

    /* Step 3: Rotate — each thread computes y[tid] = row tid of Π · x_unit */
    float y_val = 0.0f;
    for (int j = 0; j < d; j++) {
        const float * my_row = rotation + j * d;
        y_val += my_row[tid] * s_input[j] * inv_norm;
    }

    /* Step 4: Find nearest codebook centroid */
    uint8_t s_indices = tq_find_nearest(y_val, d_codebook_3, 8);

    /* Step 5: Cooperative bit-packing (3-bit) */
    /* Each thread packs its own 3 bits into the shared packed array */
    // if (tid == 0) {
    //     /* Clear output */
    //     for (int i = 0; i < TQ3_INDEX_BYTES; i++) s_packed[i] = 0;
    // }
    // __syncthreads();

    unsigned int bit[3] = {0};
    {
#pragma unroll
        for (int b = 0; b < 3; b++) {
            unsigned int val = 0;
            if (s_indices & (1 << b)) {
                // atomicOr((unsigned int *)(s_packed + (bit_pos / 8) - (bit_pos / 8) % 4),
                //          (unsigned int)(1 << (bit_pos % 32)));
                val |= (1 << lane_id);
            }
            bit[b] = __reduce_or_sync(0xffffffff, val);
        }
    }
    /* Alternative: single-threaded packing is simpler and fast enough
     * for 48 bytes. Use if atomicOr alignment is problematic. */
    // __syncthreads();

// #pragma unroll
//     for (int k = 0; k < 32; k++) {
// #pragma unroll
//         for (int i = 0; i < 3; i++) {
//             int          pos  = 3 * k + i;          // output bit index in [0, 95]
//             unsigned int b  = (bit[i] >> k) & 1u;   // extract bit k from a[i]
//             bit_inter[pos >> 5] |= b << (pos & 31);        // pos>>5 = word, pos&31 = bit within word
//         }
//     }

    // 96-bit result stored as two 64-bit halves, then split into 3×32
    // Process a[i] byte by byte, each byte spreads to 24 bits, shifted to position i
    unsigned long long lo = 0, hi = 0;  // bits [0..63] and [64..95]

    for (int i = 0; i < 3; i++) {
        for (int byte_idx = 0; byte_idx < 4; byte_idx++) {
            unsigned int byte_val = (bit[i] >> (8 * byte_idx)) & 0xFF;
            unsigned int spread   = d_spread3[byte_val];          // 24-bit spread
            int base_pos = 24 * byte_idx + i;                   // output bit start
            if (base_pos < 64)
                lo |= (unsigned long long)spread << base_pos;

            if (base_pos >= 64)
                hi |= (unsigned long long)spread << (base_pos - 64);   // was: >> (64 - base_pos) → UB
            else if (base_pos + 24 > 64)
                hi |= (unsigned long long)spread >> (64 - base_pos);   // straddle case: correct
        }
    }

    /* Step 6: Write output */
    if (tid % 32 == 0) {
        block_tq3 * blk = (block_tq3 *)((uint8_t *)dst +
                           vec_idx * sizeof(block_tq3));
        if (tid == 0) {
            blk->norm = norm;
        }
        unsigned int * blk_indices = (unsigned int *)(blk->indices + tid/32*TQ3_INDEX_BYTES/nwarps); // 12 to be generalized
        blk_indices[0] = (unsigned int)(lo);
        blk_indices[1] = (unsigned int)(lo >> 32);
        blk_indices[2] = (unsigned int)(hi);
    }
}

/* =========================================================================
 * Section 4: Dequantize Kernel
 *
 * Critical path for flash attention: KV cache read → dequantize → attention.
 * Must be as fast as possible.
 *
 * Workflow:
 *   1. Thread 0 loads norm and packed indices
 *   2. Each thread unpacks its own index
 *   3. Each thread looks up codebook centroid → y_hat[tid]
 *   4. Each thread computes x_hat[tid] = (Π^T · y_hat)[tid]
 *      = sum_j Π[j][tid] * y_hat[j]
 *   5. Scale by norm and write output
 * ========================================================================= */

__global__ void tq_dequantize_kernel_tq3(
    const void  * __restrict__ src,      /* [n_vectors × block_tq3] in  */
    float       * __restrict__ dst,      /* [n_vectors × 128] output    */
    const float * __restrict__ rotation, /* [128 × 128] rotation matrix */
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    __shared__ float s_y_hat[TQ_HEAD_DIM];
    __shared__ uint8_t s_packed[TQ3_INDEX_BYTES];
    __shared__ float s_norm;

    /* Step 1: Load block data */
    const block_tq3 * blk = (const block_tq3 *)((const uint8_t *)src +
                              vec_idx * sizeof(block_tq3));

    if (tid == 0) {
        s_norm = blk->norm;
        for (int i = 0; i < TQ3_INDEX_BYTES; i++) {
            s_packed[i] = blk->indices[i];
        }
    }
    __syncthreads();

    /* Handle zero vector */
    if (fabsf(s_norm) < 1e-15f) {
        dst[vec_idx * d + tid] = 0.0f;
        return;
    }

    /* Step 2: Each thread unpacks its own 3-bit index */
    uint8_t my_idx;
    {
        int bit_start = tid * 3;
        my_idx = 0;
        for (int b = 0; b < 3; b++) {
            int bit_pos = bit_start + b;
            if (s_packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                my_idx |= (1 << b);
            }
        }
    }

    /* Step 3: Look up codebook centroid */
    s_y_hat[tid] = d_codebook_3[my_idx];
    __syncthreads();

    /* Step 4: Rotate back — x_hat[tid] = sum_j Π^T[tid][j] * y_hat[j]
     *                                   = sum_j Π[j][tid] * y_hat[j] */
    float x_val = 0.0f;
    for (int j = 0; j < d; j++) {
        x_val += rotation[j * d + tid] * s_y_hat[j];
    }

    /* Step 5: Scale and write */
    dst[vec_idx * d + tid] = x_val * s_norm;
}

/* =========================================================================
 * Section 5: Host-side Launch Wrappers
 * ========================================================================= */

/* Initialize constant memory with codebooks (call once at startup) */
extern "C"
void tq_cuda_init_codebooks(const unsigned int *spread3) {
    cudaMemcpyToSymbol(d_codebook_3, TQ_CODEBOOK_3,
                        8 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_codebook_4, TQ_CODEBOOK_4,
                        16 * sizeof(float), 0, cudaMemcpyHostToDevice);
    cudaError_t err =cudaMemcpyToSymbol(d_spread3, spread3,
                        256 * sizeof(unsigned int), 0, cudaMemcpyHostToDevice);

}

/* Quantize n_vectors on GPU */
extern "C"
void tq_cuda_quantize_tq3(
    const float * d_src,       /* Device: [n_vectors × 128] */
    void        * d_dst,       /* Device: [n_vectors × sizeof(block_tq3)] */
    const float * d_rotation,  /* Device: [128 × 128] rotation matrix */
    int n_vectors,
    cudaStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_quantize_kernel_tq3<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors
    );
}

/* Dequantize n_vectors on GPU */
extern "C"
void tq_cuda_dequantize_tq3(
    const void  * d_src,
    float       * d_dst,
    const float * d_rotation,
    int n_vectors,
    cudaStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_dequantize_kernel_tq3<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors
    );
}

/* =========================================================================
 * Section 6: Flash Attention Integration Kernel (Fused Dequantize + Dot)
 *
 * For maximum performance, instead of dequantizing KV cache vectors to
 * FP16 and then running flash attention, we can fuse the dequantize
 * directly into the attention dot product.
 *
 * This computes: dot(Q_vec, dequant(KV_block)) without materializing
 * the full dequantized vector in memory.
 *
 * The math: Q · x_hat = Q · (||x|| · Π^T · y_hat)
 *                      = ||x|| · (Q · Π^T) · y_hat
 *                      = ||x|| · Q_rotated · y_hat
 *
 * Where Q_rotated = Q · Π^T can be precomputed once per query.
 * Then the dot product with y_hat only needs codebook lookups.
 *
 * This is the ultimate optimization — reduces the dequant+dot to:
 *   1. One codebook lookup per coordinate (register-cached)
 *   2. One multiply-accumulate per coordinate
 *   3. One scalar multiply by norm
 *
 * NOTE: This kernel is a forward-looking design for when flash attention
 * integration is ready. The non-fused path (dequant → standard FA) works
 * as the initial integration point.
 * ========================================================================= */

__global__ void tq_fused_dot_tq3(
    const float * __restrict__ q_rotated, /* [n_queries × 128] = Q · Π^T */
    const void  * __restrict__ kv_blocks, /* [n_kv × block_tq3] */
    float       * __restrict__ scores,    /* [n_queries × n_kv] output */
    int n_queries,
    int n_kv
) {
    /* Grid: (n_kv, n_queries), Block: (128) */
    const int kv_idx = blockIdx.x;
    const int q_idx  = blockIdx.y;
    if (kv_idx >= n_kv || q_idx >= n_queries) return;

    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    /* Load KV block */
    const block_tq3 * blk = (const block_tq3 *)((const uint8_t *)kv_blocks +
                              kv_idx * sizeof(block_tq3));

    __shared__ float s_norm;
    __shared__ uint8_t s_packed[TQ3_INDEX_BYTES];

    if (tid == 0) {
        s_norm = blk->norm;
        for (int i = 0; i < TQ3_INDEX_BYTES; i++) {
            s_packed[i] = blk->indices[i];
        }
    }
    __syncthreads();

    /* Unpack index for this coordinate */
    uint8_t my_idx;
    {
        int bit_start = tid * 3;
        my_idx = 0;
        for (int b = 0; b < 3; b++) {
            int bit_pos = bit_start + b;
            if (s_packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                my_idx |= (1 << b);
            }
        }
    }

    /* Lookup centroid and multiply with pre-rotated query */
    float y_hat_val = d_codebook_3[my_idx];
    float q_val = q_rotated[q_idx * d + tid];
    float partial = q_val * y_hat_val;

    /* Warp reduction for dot product */
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }

    /* Cross-warp reduction */
    __shared__ float s_warp_dots[4];
    if (tid % 32 == 0) {
        s_warp_dots[tid / 32] = partial;
    }
    __syncthreads();

    if (tid == 0) {
        float dot = s_warp_dots[0] + s_warp_dots[1] +
                    s_warp_dots[2] + s_warp_dots[3];
        scores[q_idx * n_kv + kv_idx] = dot * s_norm;
    }
}

/* Host wrapper for fused dot product */
extern "C"
void tq_cuda_fused_dot_tq3(
    const float * d_q_rotated,
    const void  * d_kv_blocks,
    float       * d_scores,
    int n_queries,
    int n_kv,
    cudaStream_t stream
) {
    dim3 grid(n_kv, n_queries);
    dim3 block(TQ_HEAD_DIM);
    tq_fused_dot_tq3<<<grid, block, 0, stream>>>(
        d_q_rotated, d_kv_blocks, d_scores, n_queries, n_kv
    );
}
