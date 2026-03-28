/*
 * TurboQuant: Near-Optimal KV Cache Quantization for ik_llama.cpp
 * ================================================================
 * Reference: Zandieh et al., ICLR 2026 (arXiv:2504.19874)
 * 
 * Implements PolarQuant (Algorithm 1) + QJL error correction for
 * KV cache compression to 3-3.5 bits per value with near-zero
 * accuracy loss.
 *
 * Built for Nexus Grove (ng-01) — designed to integrate into
 * ik_llama.cpp's existing KV cache quantization pipeline alongside
 * the existing q4_0, q8_0, etc. types.
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */

#ifndef GGML_TURBOQUANT_H
#define GGML_TURBOQUANT_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * Section 1: Configuration Constants
 * ========================================================================= */

/* Head dimension — standard for modern transformers (Qwen, Llama, etc.) */
#define TQ_HEAD_DIM 128

/* Supported bit-widths */
#define TQ_BITS_2  2
#define TQ_BITS_3  3
#define TQ_BITS_4  4

/* Default operational bit-width (paper's quality-neutral sweet spot) */
#define TQ_DEFAULT_BITS  3

/* Number of quantization levels per bit-width */
#define TQ_LEVELS_2  4
#define TQ_LEVELS_3  8
#define TQ_LEVELS_4  16

/* Rotation matrix seed — deterministic for reproducibility across sessions */
#define TQ_ROTATION_SEED  42

/* =========================================================================
 * Section 2: Pre-computed Lloyd-Max Codebooks (from turboquant_codebooks.json)
 *
 * These are optimal scalar quantizer centroids for the Beta distribution
 * induced by random rotation of unit vectors in R^128.
 * Computed via Lloyd-Max algorithm per Theorem 1 of the paper.
 * ========================================================================= */

/* 2-bit codebook: 4 centroids */
static const float TQ_CODEBOOK_2[4] = {
    -0.13311451677280386f,
    -0.04002746648341520f,
     0.04002746648341517f,
     0.13311451677280380f
};

/* 3-bit codebook: 8 centroids */
static const float TQ_CODEBOOK_3[8] = {
    -0.18904037194348838f,
    -0.11879501670185091f,
    -0.06702922184405663f,
    -0.02174971334976657f,
     0.02174971334976654f,
     0.06702922184405660f,
     0.11879501670185087f,
     0.18904037194348833f
};

/* 4-bit codebook: 16 centroids */
static const float TQ_CODEBOOK_4[16] = {
    -0.23961253307138700f,
    -0.18317108415643454f,
    -0.14430970076906538f,
    -0.11276586366299288f,
    -0.08507481024405737f,
    -0.05962130616889217f,
    -0.03539017687270855f,
    -0.01173284981923122f,
     0.01173284981923120f,
     0.03539017687270851f,
     0.05962130616889214f,
     0.08507481024405730f,
     0.11276586366299284f,
     0.14430970076906535f,
     0.18317108415643450f,
     0.23961253307138697f
};

/* =========================================================================
 * Section 3: Data Structures
 * ========================================================================= */

/*
 * TurboQuant quantized block — stores one quantized head vector.
 *
 * Memory layout for TQ3 (3-bit, d=128):
 *   - norm:    4 bytes  (float32, original L2 norm)
 *   - indices: 48 bytes (128 values × 3 bits = 384 bits = 48 bytes)
 *   Total: 52 bytes per vector
 *
 * Compare to FP16: 128 × 2 = 256 bytes per vector → 4.9x compression
 *
 * The indices are bit-packed: for b-bit quantization, each coordinate
 * index (0 to 2^b - 1) is stored in exactly b bits, packed sequentially.
 */

/* Block size for TQ3: one head_dim vector */
#define TQ3_BLOCK_SIZE    TQ_HEAD_DIM
#define TQ3_BITS_PER_VAL  3
#define TQ3_INDEX_BYTES   ((TQ3_BLOCK_SIZE * TQ3_BITS_PER_VAL + 7) / 8)  /* 48 */

typedef struct {
    float    norm;                      /* Original L2 norm of the vector    */
    uint8_t  indices[TQ3_INDEX_BYTES];  /* Bit-packed codebook indices       */
} block_tq3;

/* Verify size at compile time */
_Static_assert(sizeof(block_tq3) == 4 + TQ3_INDEX_BYTES,
               "block_tq3 size mismatch");

/* Block size for TQ4: one head_dim vector */
#define TQ4_BLOCK_SIZE    TQ_HEAD_DIM
#define TQ4_BITS_PER_VAL  4
#define TQ4_INDEX_BYTES   ((TQ4_BLOCK_SIZE * TQ4_BITS_PER_VAL + 7) / 8)  /* 64 */

typedef struct {
    float    norm;
    uint8_t  indices[TQ4_INDEX_BYTES];
} block_tq4;

_Static_assert(sizeof(block_tq4) == 4 + TQ4_INDEX_BYTES,
               "block_tq4 size mismatch");

/*
 * Rotation matrix context — generated once at KV cache init,
 * reused for all quantize/dequantize operations.
 *
 * The matrix is d×d orthogonal, generated via QR decomposition
 * of a seeded random Gaussian matrix (Algorithm 1, Line 2).
 *
 * For d=128, this is 128×128×4 = 64 KB per rotation context.
 * Two contexts are needed (one for K cache, one for V cache) = 128 KB total.
 * Negligible compared to the KV cache itself.
 */
typedef struct {
    int      d;                             /* Dimension (TQ_HEAD_DIM)       */
    int      bits;                          /* Bit-width (2, 3, or 4)        */
    int      n_levels;                      /* 2^bits                        */
    const float * codebook;                 /* Pointer to static codebook    */
    float    rotation[TQ_HEAD_DIM * TQ_HEAD_DIM]; /* Orthogonal rotation Π  */
} tq_context;

/* =========================================================================
 * Section 4: Core API — CPU Reference Implementation
 * ========================================================================= */

/*
 * Initialize a TurboQuant context with a given bit-width and seed.
 * Generates the rotation matrix via QR decomposition of seeded Gaussian.
 *
 * @param ctx   Output context
 * @param bits  Bit-width (2, 3, or 4)
 * @param seed  RNG seed for rotation matrix (use TQ_ROTATION_SEED)
 * @return 0 on success, -1 on error
 */
int tq_context_init(tq_context * ctx, int bits, uint64_t seed);

/*
 * Quantize a single head-dimension vector (Algorithm 1).
 *
 * Steps:
 *   1. Store ||x||_2 as norm
 *   2. Normalize: x_unit = x / ||x||
 *   3. Rotate: y = Π · x_unit
 *   4. For each y_j, find nearest codebook centroid index
 *   5. Bit-pack indices into output block
 *
 * @param ctx     Initialized TQ context
 * @param src     Input vector, float[TQ_HEAD_DIM]
 * @param dst     Output block (block_tq3 or block_tq4, cast to void*)
 */
void tq_quantize(const tq_context * ctx, const float * src, void * dst);

/*
 * Dequantize a single block back to float vector (Algorithm 1 inverse).
 *
 * Steps:
 *   1. Unpack bit-packed indices
 *   2. Map indices to codebook centroids: y_hat_j = codebook[idx_j]
 *   3. Rotate back: x_hat = Π^T · y_hat
 *   4. Scale by stored norm
 *
 * @param ctx     Initialized TQ context
 * @param src     Input block (block_tq3 or block_tq4, cast to const void*)
 * @param dst     Output vector, float[TQ_HEAD_DIM]
 */
void tq_dequantize(const tq_context * ctx, const void * src, float * dst);

/*
 * Quantize a batch of vectors (e.g., all KV heads for one token in one layer).
 *
 * @param ctx        Initialized TQ context
 * @param src        Input: n_vectors × TQ_HEAD_DIM floats (row-major)
 * @param dst        Output: n_vectors × sizeof(block_tqN) bytes
 * @param n_vectors  Number of vectors to quantize
 */
void tq_quantize_batch(const tq_context * ctx, const float * src,
                        void * dst, int n_vectors);

/*
 * Dequantize a batch of vectors.
 */
void tq_dequantize_batch(const tq_context * ctx, const void * src,
                          float * dst, int n_vectors);

/* =========================================================================
 * Section 5: Utility Functions
 * ========================================================================= */

/*
 * Returns the size in bytes of one quantized block for the given bit-width.
 */
static inline size_t tq_block_size(int bits) {
    switch (bits) {
        case 3:  return sizeof(block_tq3);
        case 4:  return sizeof(block_tq4);
        default: return 0;
    }
}

/*
 * Returns the compression ratio vs FP16 for the given bit-width.
 */
static inline float tq_compression_ratio(int bits) {
    size_t fp16_size = TQ_HEAD_DIM * 2;  /* 256 bytes */
    size_t tq_size   = tq_block_size(bits);
    if (tq_size == 0) return 0.0f;
    return (float)fp16_size / (float)tq_size;
}

/* =========================================================================
 * Section 6: Bit-packing Helpers
 * ========================================================================= */

/*
 * Pack an array of b-bit indices into a byte array.
 * indices[i] must be in range [0, 2^b - 1].
 */
void tq_pack_indices(const uint8_t * indices, uint8_t * packed,
                     int n_values, int bits);

/*
 * Unpack a byte array into an array of b-bit indices.
 */
void tq_unpack_indices(const uint8_t * packed, uint8_t * indices,
                       int n_values, int bits);

#ifdef __cplusplus
}
#endif

#endif /* GGML_TURBOQUANT_H */
