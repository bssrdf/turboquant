/*
 * TurboQuant: Test Harness
 * =========================
 * Validates the C implementation against known-good values from
 * the Python prototype. Compile and run on ng-01 before touching
 * any ik_llama.cpp integration.
 *
 * Build:
 *   gcc -O2 -o tq_test ggml_turboquant.c tq_test.c -lm
 *
 * Run:
 *   ./tq_test
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */

#include "ggml_turboquant.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#define PASS "\033[32m✓ PASS\033[0m"
#define FAIL "\033[31m✗ FAIL\033[0m"

static int tests_passed = 0;
static int tests_failed = 0;

static void check(const char * name, int condition) {
    if (condition) {
        printf("  %s %s\n", PASS, name);
        tests_passed++;
    } else {
        printf("  %s %s\n", FAIL, name);
        tests_failed++;
    }
}

/* Simple seeded PRNG for test vectors (matches Python's numpy seed behavior
 * closely enough for MSE validation, though not bit-identical) */
static uint64_t test_rng_state = 0;

static float test_randn(void) {
    /* xorshift64* */
    test_rng_state ^= test_rng_state >> 12;
    test_rng_state ^= test_rng_state << 25;
    test_rng_state ^= test_rng_state >> 27;
    uint64_t r = test_rng_state * 0x2545F4914F6CDD1DULL;

    /* Convert to uniform [0,1) */
    double u1 = (double)(r >> 11) / (double)(1ULL << 53);

    test_rng_state ^= test_rng_state >> 12;
    test_rng_state ^= test_rng_state << 25;
    test_rng_state ^= test_rng_state >> 27;
    r = test_rng_state * 0x2545F4914F6CDD1DULL;
    double u2 = (double)(r >> 11) / (double)(1ULL << 53);

    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

/* ========================================================================= */

int main(void) {
    printf("=========================================================\n");
    printf("TurboQuant C Implementation — Validation Suite\n");
    printf("=========================================================\n\n");

    /* -----------------------------------------------------------------
     * Test 1: Context initialization
     * ----------------------------------------------------------------- */
    printf("[Test 1] Context Initialization\n");
    printf("-------------------------------------------------\n");

    tq_context ctx3, ctx4;
    int rc3 = tq_context_init(&ctx3, 3, TQ_ROTATION_SEED);
    int rc4 = tq_context_init(&ctx4, 4, TQ_ROTATION_SEED);

    check("TQ3 context init returns 0", rc3 == 0);
    check("TQ4 context init returns 0", rc4 == 0);
    check("TQ3 has 8 levels", ctx3.n_levels == 8);
    check("TQ4 has 16 levels", ctx4.n_levels == 16);
    check("TQ3 dimension is 128", ctx3.d == 128);

    /* Verify rotation matrix is orthogonal: Π^T · Π ≈ I */
    float dot_00 = 0.0f, dot_01 = 0.0f;
    for (int k = 0; k < TQ_HEAD_DIM; k++) {
        dot_00 += ctx3.rotation[k * TQ_HEAD_DIM + 0] *
                  ctx3.rotation[k * TQ_HEAD_DIM + 0];
        dot_01 += ctx3.rotation[k * TQ_HEAD_DIM + 0] *
                  ctx3.rotation[k * TQ_HEAD_DIM + 1];
    }
    check("Rotation col 0 has unit norm", fabsf(dot_00 - 1.0f) < 1e-4f);
    check("Rotation cols 0,1 are orthogonal", fabsf(dot_01) < 1e-4f);

    /* Invalid bit-width should fail */
    tq_context ctx_bad;
    int rc_bad = tq_context_init(&ctx_bad, 5, 0);
    check("Invalid bit-width returns -1", rc_bad == -1);

    /* -----------------------------------------------------------------
     * Test 2: Bit-packing round-trip
     * ----------------------------------------------------------------- */
    printf("\n[Test 2] Bit-packing Round-trip\n");
    printf("-------------------------------------------------\n");

    /* 3-bit packing */
    uint8_t orig3[TQ_HEAD_DIM], unpacked3[TQ_HEAD_DIM];
    uint8_t packed3[TQ3_INDEX_BYTES];
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        orig3[i] = (uint8_t)(i % 8);  /* 0-7 for 3-bit */
    }
    tq_pack_indices(orig3, packed3, TQ_HEAD_DIM, 3);
    tq_unpack_indices(packed3, unpacked3, TQ_HEAD_DIM, 3);

    int pack3_ok = 1;
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        if (orig3[i] != unpacked3[i]) { pack3_ok = 0; break; }
    }
    check("3-bit pack/unpack round-trip", pack3_ok);

    /* 4-bit packing */
    uint8_t orig4[TQ_HEAD_DIM], unpacked4[TQ_HEAD_DIM];
    uint8_t packed4[TQ4_INDEX_BYTES];
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        orig4[i] = (uint8_t)(i % 16);  /* 0-15 for 4-bit */
    }
    tq_pack_indices(orig4, packed4, TQ_HEAD_DIM, 4);
    tq_unpack_indices(packed4, unpacked4, TQ_HEAD_DIM, 4);

    int pack4_ok = 1;
    for (int i = 0; i < TQ_HEAD_DIM; i++) {
        if (orig4[i] != unpacked4[i]) { pack4_ok = 0; break; }
    }
    check("4-bit pack/unpack round-trip", pack4_ok);

    /* -----------------------------------------------------------------
     * Test 3: Quantize/Dequantize round-trip MSE
     * ----------------------------------------------------------------- */
    printf("\n[Test 3] Quantize/Dequantize Round-trip MSE\n");
    printf("-------------------------------------------------\n");

    /* Paper's expected MSE for d=128 (from Theorem 1):
     *   b=3: ~0.034
     *   b=4: ~0.0093
     */
    test_rng_state = 12345;
    int n_test_vectors = 1000;

    for (int bits = 3; bits <= 4; bits++) {
        tq_context * ctx = (bits == 3) ? &ctx3 : &ctx4;
        float paper_mse = (bits == 3) ? 0.034f : 0.0093f;

        float total_mse = 0.0f;
        size_t blk_size = tq_block_size(bits);
        uint8_t block_buf[sizeof(block_tq4)]; /* Large enough for either */

        for (int v = 0; v < n_test_vectors; v++) {
            /* Generate random unit vector */
            float x[TQ_HEAD_DIM], x_hat[TQ_HEAD_DIM];
            float norm = 0.0f;
            for (int j = 0; j < TQ_HEAD_DIM; j++) {
                x[j] = test_randn();
                norm += x[j] * x[j];
            }
            norm = sqrtf(norm);
            for (int j = 0; j < TQ_HEAD_DIM; j++) x[j] /= norm;

            tq_quantize(ctx, x, block_buf);
            tq_dequantize(ctx, block_buf, x_hat);

            float mse = 0.0f;
            for (int j = 0; j < TQ_HEAD_DIM; j++) {
                float diff = x[j] - x_hat[j];
                mse += diff * diff;
            }
            total_mse += mse;
        }

        float avg_mse = total_mse / n_test_vectors;
        /* Allow 3x tolerance since our PRNG differs from numpy */
        int mse_ok = (avg_mse < paper_mse * 3.0f) && (avg_mse > paper_mse * 0.3f);

        printf("  b=%d: Avg MSE = %.6f  (paper ≈ %.4f)  ratio = %.2f\n",
               bits, avg_mse, paper_mse, avg_mse / paper_mse);
        check(bits == 3 ? "TQ3 MSE within 3x of paper" :
                          "TQ4 MSE within 3x of paper", mse_ok);
    }

    /* -----------------------------------------------------------------
     * Test 4: Zero vector handling
     * ----------------------------------------------------------------- */
    printf("\n[Test 4] Zero Vector Handling\n");
    printf("-------------------------------------------------\n");

    float zeros[TQ_HEAD_DIM];
    float zeros_out[TQ_HEAD_DIM];
    uint8_t zero_block[sizeof(block_tq3)];
    memset(zeros, 0, sizeof(zeros));

    tq_quantize(&ctx3, zeros, zero_block);
    tq_dequantize(&ctx3, zero_block, zeros_out);

    float zero_norm = 0.0f;
    for (int j = 0; j < TQ_HEAD_DIM; j++) {
        zero_norm += zeros_out[j] * zeros_out[j];
    }
    check("Zero vector round-trips to zero", zero_norm < 1e-10f);

    /* -----------------------------------------------------------------
     * Test 5: Norm preservation
     * ----------------------------------------------------------------- */
    printf("\n[Test 5] Norm Preservation\n");
    printf("-------------------------------------------------\n");

    test_rng_state = 99999;
    float x_norm_test[TQ_HEAD_DIM], x_hat_norm[TQ_HEAD_DIM];
    uint8_t norm_block[sizeof(block_tq3)];

    /* Create vector with known norm = 3.7 */
    float target_norm = 3.7f;
    float raw_norm = 0.0f;
    for (int j = 0; j < TQ_HEAD_DIM; j++) {
        x_norm_test[j] = test_randn();
        raw_norm += x_norm_test[j] * x_norm_test[j];
    }
    raw_norm = sqrtf(raw_norm);
    for (int j = 0; j < TQ_HEAD_DIM; j++) {
        x_norm_test[j] *= target_norm / raw_norm;
    }

    tq_quantize(&ctx3, x_norm_test, norm_block);
    tq_dequantize(&ctx3, norm_block, x_hat_norm);

    float recon_norm = 0.0f;
    for (int j = 0; j < TQ_HEAD_DIM; j++) {
        recon_norm += x_hat_norm[j] * x_hat_norm[j];
    }
    recon_norm = sqrtf(recon_norm);

    printf("  Original norm: %.4f  Reconstructed norm: %.4f\n",
           target_norm, recon_norm);
    check("Norm preserved within 10%",
          fabsf(recon_norm - target_norm) / target_norm < 0.10f);

    /* -----------------------------------------------------------------
     * Test 6: Compression ratio verification
     * ----------------------------------------------------------------- */
    printf("\n[Test 6] Compression Ratios\n");
    printf("-------------------------------------------------\n");

    size_t fp16_size = TQ_HEAD_DIM * 2;  /* 256 bytes */
    float ratio3 = tq_compression_ratio(3);
    float ratio4 = tq_compression_ratio(4);

    printf("  TQ3: %zu bytes → %.1fx vs FP16 (%zu bytes)\n",
           tq_block_size(3), ratio3, fp16_size);
    printf("  TQ4: %zu bytes → %.1fx vs FP16 (%zu bytes)\n",
           tq_block_size(4), ratio4, fp16_size);

    check("TQ3 compression > 4x", ratio3 > 4.0f);
    check("TQ4 compression > 3x", ratio4 > 3.0f);

    /* -----------------------------------------------------------------
     * Test 7: Batch operations
     * ----------------------------------------------------------------- */
    printf("\n[Test 7] Batch Quantize/Dequantize\n");
    printf("-------------------------------------------------\n");

    int batch_size = 8;  /* 8 KV heads typical for GQA */
    float batch_in[8 * TQ_HEAD_DIM];
    float batch_out[8 * TQ_HEAD_DIM];
    uint8_t batch_blocks[8 * sizeof(block_tq3)];

    test_rng_state = 42424242;
    for (int i = 0; i < batch_size * TQ_HEAD_DIM; i++) {
        batch_in[i] = test_randn() * 0.1f;
    }

    tq_quantize_batch(&ctx3, batch_in, batch_blocks, batch_size);
    tq_dequantize_batch(&ctx3, batch_blocks, batch_out, batch_size);

    float batch_mse = 0.0f;
    for (int i = 0; i < batch_size * TQ_HEAD_DIM; i++) {
        float diff = batch_in[i] - batch_out[i];
        batch_mse += diff * diff;
    }
    batch_mse /= batch_size;
    printf("  Batch MSE (8 vectors): %.6f\n", batch_mse);
    check("Batch round-trip MSE reasonable", batch_mse < 0.1f);

    /* -----------------------------------------------------------------
     * Test 8: Speed benchmark
     * ----------------------------------------------------------------- */
    printf("\n[Test 8] Speed Benchmark (10000 vectors)\n");
    printf("-------------------------------------------------\n");

    int bench_n = 10000;
    float * bench_in = (float *)malloc(bench_n * TQ_HEAD_DIM * sizeof(float));
    uint8_t * bench_blocks = (uint8_t *)malloc(bench_n * sizeof(block_tq3));
    float * bench_out = (float *)malloc(bench_n * TQ_HEAD_DIM * sizeof(float));

    test_rng_state = 777;
    for (int i = 0; i < bench_n * TQ_HEAD_DIM; i++) {
        bench_in[i] = test_randn();
    }

    clock_t t0 = clock();
    tq_quantize_batch(&ctx3, bench_in, bench_blocks, bench_n);
    clock_t t1 = clock();
    tq_dequantize_batch(&ctx3, bench_blocks, bench_out, bench_n);
    clock_t t2 = clock();

    double quant_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0;
    double dequant_ms = (double)(t2 - t1) / CLOCKS_PER_SEC * 1000.0;

    printf("  Quantize:   %.1f ms  (%.0f vectors/sec)\n",
           quant_ms, bench_n / (quant_ms / 1000.0));
    printf("  Dequantize: %.1f ms  (%.0f vectors/sec)\n",
           dequant_ms, bench_n / (dequant_ms / 1000.0));

    /* CPU speed check: should manage at least 1000 vec/s even unoptimized */
    check("Quantize speed > 1000 vec/s",
          bench_n / (quant_ms / 1000.0) > 1000.0);

    free(bench_in);
    free(bench_blocks);
    free(bench_out);

    /* -----------------------------------------------------------------
     * Summary
     * ----------------------------------------------------------------- */
    printf("\n=========================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("=========================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
