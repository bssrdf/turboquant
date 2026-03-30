/*
 * TurboQuant: CUDA API Declarations
 * ==================================
 * GPU-accelerated quantize/dequantize kernels for KV cache vectors.
 *
 * This header provides the CUDA-specific API that wraps the device kernels
 * implemented in ggml_turboquant.cu.
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */

#ifndef GGML_TURBOQUANT_CUDA_H
#define GGML_TURBOQUANT_CUDA_H

#include "ggml_turboquant.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * CUDA Initialization
 * ========================================================================= */

/**
 * Initialize CUDA constant memory with codebooks.
 * 
 * This function copies the TQ3 and TQ4 codebooks from host to device
 * constant memory. Must be called once at program startup before using
 * any TurboQuant CUDA kernels.
 *
 * @return void
 */
void tq_cuda_init_codebooks(const unsigned int *);

/* =========================================================================
 * TQ3 (3-bit) Kernels
 * ========================================================================= */

/**
 * Quantize n_vectors on GPU using TQ3 (3-bit per value).
 *
 * Each input vector is:
 *   1. Normalized to unit length (norm stored)
 *   2. Rotated by orthogonal matrix Π
 *   3. Each coordinate quantized to nearest codebook centroid
 *   4. Indices bit-packed into block_tq3 structure
 *
 * @param d_src      Device pointer: [n_vectors × 128] FP32 input vectors
 * @param d_dst      Device pointer: [n_vectors × sizeof(block_tq3)] output blocks
 * @param d_rotation Device pointer: [128 × 128] orthogonal rotation matrix Π
 * @param n_vectors  Number of vectors to quantize
 * @param stream     CUDA stream for async execution (use 0 for default stream)
 */
void tq_cuda_quantize_tq3(
    const float * d_src,
    void        * d_dst,
    const float * d_rotation,
    int n_vectors,
    cudaStream_t stream
);

/**
 * Dequantize n_vectors on GPU using TQ3 (3-bit per value).
 *
 * Each quantized block is:
 *   1. Unpacked from bit-packed indices
 *   2. Indices mapped to codebook centroids → y_hat
 *   3. Rotated back: x_hat = Π^T · y_hat
 *   4. Scaled by stored norm
 *
 * This is the critical path for flash attention: KV cache read → dequantize → attention.
 *
 * @param d_src      Device pointer: [n_vectors × sizeof(block_tq3)] input blocks
 * @param d_dst      Device pointer: [n_vectors × 128] FP32 output vectors
 * @param d_rotation Device pointer: [128 × 128] orthogonal rotation matrix Π
 * @param n_vectors  Number of vectors to dequantize
 * @param stream     CUDA stream for async execution (use 0 for default stream)
 */
void tq_cuda_dequantize_tq3(
    const void  * d_src,
    float       * d_dst,
    const float * d_rotation,
    int n_vectors,
    cudaStream_t stream
);

/* =========================================================================
 * Fused Kernels (Flash Attention Integration)
 * ========================================================================= */

/**
 * Fused dequantize + dot product for flash attention.
 *
 * Computes: scores[i,j] = dot(Q_rotated[i], dequant(kv_blocks[j]))
 *
 * Without materializing the full dequantized KV vectors in memory.
 * This is the ultimate optimization for attention with quantized KV cache.
 *
 * The query vectors must be pre-rotated: Q_rotated = Q · Π^T
 * This can be done once per query token before the attention loop.
 *
 * @param d_q_rotated Device pointer: [n_queries × 128] pre-rotated queries (Q · Π^T)
 * @param d_kv_blocks Device pointer: [n_kv × sizeof(block_tq3)] quantized KV blocks
 * @param d_scores    Device pointer: [n_queries × n_kv] output attention scores
 * @param n_queries   Number of query vectors
 * @param n_kv        Number of KV blocks
 * @param stream      CUDA stream for async execution
 */
void tq_cuda_fused_dot_tq3(
    const float * d_q_rotated,
    const void  * d_kv_blocks,
    float       * d_scores,
    int n_queries,
    int n_kv,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBOQUANT_CUDA_H */
