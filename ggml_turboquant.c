/*
 * TurboQuant: CPU Reference Implementation
 * ==========================================
 * Implements Algorithm 1 (TurboQuant_mse) from Zandieh et al., ICLR 2026.
 *
 * This is the portable C implementation that runs on CPU.
 * The CUDA implementation (ggml_turboquant.cu) mirrors this logic
 * with GPU-optimized kernels.
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */

#include "ggml_turboquant.h"
#include <math.h>
#include <string.h>
#include <float.h>

/* =========================================================================
 * Section 1: Seeded RNG (xoshiro256** for reproducible rotation matrices)
 *
 * We need a portable, high-quality PRNG that produces identical sequences
 * across platforms given the same seed. xoshiro256** fits perfectly.
 * ========================================================================= */

typedef struct {
    uint64_t s[4];
} tq_rng;

static inline uint64_t tq_rng_rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t tq_rng_next(tq_rng * rng) {
    const uint64_t result = tq_rng_rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = tq_rng_rotl(rng->s[3], 45);
    return result;
}

static void tq_rng_seed(tq_rng * rng, uint64_t seed) {
    /* SplitMix64 to expand a single seed into 4 state words */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

/* Convert uint64 to standard normal via Box-Muller */
static float tq_rng_normal(tq_rng * rng) {
    /* Generate two uniform [0,1) values */
    double u1 = (double)(tq_rng_next(rng) >> 11) / (double)(1ULL << 53);
    double u2 = (double)(tq_rng_next(rng) >> 11) / (double)(1ULL << 53);
    /* Clamp to avoid log(0) */
    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

/* =========================================================================
 * Section 2: QR Decomposition (Modified Gram-Schmidt)
 *
 * Generates the random orthogonal rotation matrix Π from Algorithm 1, Line 2.
 * We fill a d×d matrix with standard normals, then orthogonalize via
 * modified Gram-Schmidt. The result is stored row-major in ctx->rotation.
 * ========================================================================= */

static void tq_generate_rotation(float * R, int d, uint64_t seed) {
    tq_rng rng;
    tq_rng_seed(&rng, seed);

    /* Fill with standard normals (row-major: R[i*d + j]) */
    for (int i = 0; i < d * d; i++) {
        R[i] = tq_rng_normal(&rng);
    }

    /* Modified Gram-Schmidt orthogonalization (column-wise) */
    /* Column j of R is at R[row*d + j] for row in [0, d) */
    for (int j = 0; j < d; j++) {
        /* Subtract projections of column j onto all previous columns */
        for (int k = 0; k < j; k++) {
            float dot = 0.0f;
            float norm_k_sq = 0.0f;
            for (int i = 0; i < d; i++) {
                dot      += R[i * d + j] * R[i * d + k];
                norm_k_sq += R[i * d + k] * R[i * d + k];
            }
            if (norm_k_sq > 1e-15f) {
                float scale = dot / norm_k_sq;
                for (int i = 0; i < d; i++) {
                    R[i * d + j] -= scale * R[i * d + k];
                }
            }
        }

        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += R[i * d + j] * R[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-15f) {
            float inv_norm = 1.0f / norm;
            for (int i = 0; i < d; i++) {
                R[i * d + j] *= inv_norm;
            }
        }
    }
}

/* =========================================================================
 * Section 3: Context Initialization
 * ========================================================================= */

int tq_context_init(tq_context * ctx, int bits, uint64_t seed) {
    if (!ctx) return -1;
    if (bits < 2 || bits > 4) return -1;

    ctx->d = TQ_HEAD_DIM;
    ctx->bits = bits;
    ctx->n_levels = 1 << bits;

    /* Select codebook */
    switch (bits) {
        case 2: ctx->codebook = TQ_CODEBOOK_2; break;
        case 3: ctx->codebook = TQ_CODEBOOK_3; break;
        case 4: ctx->codebook = TQ_CODEBOOK_4; break;
        default: return -1;
    }

    /* Generate rotation matrix */
    tq_generate_rotation(ctx->rotation, ctx->d, seed);

    return 0;
}

/* =========================================================================
 * Section 4: Bit-packing
 *
 * Pack/unpack arrays of b-bit values into/from byte arrays.
 * For b=3: 128 values × 3 bits = 384 bits = 48 bytes
 * For b=4: 128 values × 4 bits = 512 bits = 64 bytes
 * ========================================================================= */

void tq_pack_indices(const uint8_t * indices, uint8_t * packed,
                     int n_values, int bits) {
    memset(packed, 0, (n_values * bits + 7) / 8);

    int bit_pos = 0;
    for (int i = 0; i < n_values; i++) {
        uint8_t val = indices[i];
        for (int b = 0; b < bits; b++) {
            if (val & (1 << b)) {
                packed[bit_pos / 8] |= (1 << (bit_pos % 8));
            }
            bit_pos++;
        }
    }
}

void tq_unpack_indices(const uint8_t * packed, uint8_t * indices,
                       int n_values, int bits) {
    int bit_pos = 0;
    uint8_t mask = (1 << bits) - 1;

    for (int i = 0; i < n_values; i++) {
        uint8_t val = 0;
        for (int b = 0; b < bits; b++) {
            if (packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                val |= (1 << b);
            }
            bit_pos++;
        }
        indices[i] = val & mask;
    }
}

/* =========================================================================
 * Section 5: Quantize — Algorithm 1 (TurboQuant_mse)
 * ========================================================================= */

void tq_quantize(const tq_context * ctx, const float * src, void * dst) {
    const int d = ctx->d;
    const int bits = ctx->bits;
    const int n_levels = ctx->n_levels;
    const float * codebook = ctx->codebook;
    const float * R = ctx->rotation;

    /* Step 1: Compute L2 norm */
    float norm = 0.0f;
    for (int i = 0; i < d; i++) {
        norm += src[i] * src[i];
    }
    norm = sqrtf(norm);

    /* Temporary buffers (stack-allocated, d=128 so this is fine) */
    float y[TQ_HEAD_DIM];       /* Rotated vector */
    uint8_t indices[TQ_HEAD_DIM]; /* Codebook indices */

    if (norm < 1e-15f) {
        /* Zero vector — store zero norm and zero indices */
        if (bits == 3) {
            block_tq3 * blk = (block_tq3 *)dst;
            blk->norm = 0.0f;
            memset(blk->indices, 0, TQ3_INDEX_BYTES);
        } else if (bits == 4) {
            block_tq4 * blk = (block_tq4 *)dst;
            blk->norm = 0.0f;
            memset(blk->indices, 0, TQ4_INDEX_BYTES);
        }
        return;
    }

    float inv_norm = 1.0f / norm;

    /* Step 2-3: Normalize and rotate: y = Π · (x / ||x||) */
    /* R is row-major: R[i*d + j] = element (i,j) */
    /* y[i] = sum_j R[i*d + j] * x_unit[j] */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        const float * row = R + i * d;
        for (int j = 0; j < d; j++) {
            sum += row[j] * src[j] * inv_norm;
        }
        y[i] = sum;
    }

    /* Step 4: Find nearest codebook centroid for each coordinate */
    for (int i = 0; i < d; i++) {
        float best_dist = FLT_MAX;
        uint8_t best_idx = 0;
        for (int c = 0; c < n_levels; c++) {
            float dist = (y[i] - codebook[c]) * (y[i] - codebook[c]);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = (uint8_t)c;
            }
        }
        indices[i] = best_idx;
    }

    /* Step 5: Pack into output block */
    if (bits == 3) {
        block_tq3 * blk = (block_tq3 *)dst;
        blk->norm = norm;
        tq_pack_indices(indices, blk->indices, d, bits);
    } else if (bits == 4) {
        block_tq4 * blk = (block_tq4 *)dst;
        blk->norm = norm;
        tq_pack_indices(indices, blk->indices, d, bits);
    }
}

/* =========================================================================
 * Section 6: Dequantize — Algorithm 1 Inverse
 * ========================================================================= */

void tq_dequantize(const tq_context * ctx, const void * src, float * dst) {
    const int d = ctx->d;
    const int bits = ctx->bits;
    const float * codebook = ctx->codebook;
    const float * R = ctx->rotation;

    float norm;
    const uint8_t * packed;

    if (bits == 3) {
        const block_tq3 * blk = (const block_tq3 *)src;
        norm = blk->norm;
        packed = blk->indices;
    } else if (bits == 4) {
        const block_tq4 * blk = (const block_tq4 *)src;
        norm = blk->norm;
        packed = blk->indices;
    } else {
        memset(dst, 0, d * sizeof(float));
        return;
    }

    if (fabsf(norm) < 1e-15f) {
        memset(dst, 0, d * sizeof(float));
        return;
    }

    /* Step 1: Unpack indices */
    uint8_t indices[TQ_HEAD_DIM];
    tq_unpack_indices(packed, indices, d, bits);

    /* Step 2: Map indices to centroid values -> y_hat */
    float y_hat[TQ_HEAD_DIM];
    for (int i = 0; i < d; i++) {
        y_hat[i] = codebook[indices[i]];
    }

    /* Step 3: Rotate back: x_hat = Π^T · y_hat */
    /* Π^T row i, col j = R[j*d + i] (transpose of row-major R) */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += R[j * d + i] * y_hat[j];
        }
        /* Step 4: Scale by original norm */
        dst[i] = sum * norm;
    }
}

/* =========================================================================
 * Section 7: Batch Operations
 * ========================================================================= */

void tq_quantize_batch(const tq_context * ctx, const float * src,
                        void * dst, int n_vectors) {
    const int d = ctx->d;
    size_t blk_size = tq_block_size(ctx->bits);

    for (int v = 0; v < n_vectors; v++) {
        tq_quantize(ctx, src + v * d, (uint8_t *)dst + v * blk_size);
    }
}

void tq_dequantize_batch(const tq_context * ctx, const void * src,
                          float * dst, int n_vectors) {
    const int d = ctx->d;
    size_t blk_size = tq_block_size(ctx->bits);

    for (int v = 0; v < n_vectors; v++) {
        tq_dequantize(ctx, (const uint8_t *)src + v * blk_size, dst + v * d);
    }
}
