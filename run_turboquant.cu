/*
 * TurboQuant: CUDA Test Runner
 * =============================
 * Demonstrates how to prepare variables and call the TQ3 quantize/dequantize
 * CUDA kernels. This is a standalone test that can be compiled and run to
 * verify the GPU implementation works correctly.
 *
 * Build:
 *   nvcc -o run_turboquant run_turboquant.cu ggml_turboquant.cu -lm
 *
 * Run:
 *   ./run_turboquant
 *
 * Authors: Jim Sullivan / Claude collaboration
 * Date: 2026-03-25
 */

#include "ggml_turboquant.h"
#include "ggml_turboquant_cuda.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define PASS "\033[32m✓ PASS\033[0m"
#define FAIL "\033[31m✗ FAIL\033[0m"

/* Simple error checking for CUDA calls */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* =========================================================================
 * Helper: Generate random Gaussian vectors (matches test harness)
 * ========================================================================= */

static uint64_t rng_state = 12345;

static float randn(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    uint64_t r = rng_state * 0x2545F4914F6CDD1DULL;
    double u1 = (double)(r >> 11) / (double)(1ULL << 53);

    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    r = rng_state * 0x2545F4914F6CDD1DULL;
    double u2 = (double)(r >> 11) / (double)(1ULL << 53);

    if (u1 < 1e-15) u1 = 1e-15;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2));
}

/* =========================================================================
 * Main: Prepare variables and call kernels
 * ========================================================================= */

int main(void) {
    printf("=========================================================\n");
    printf("TurboQuant CUDA Kernel Test\n");
    printf("=========================================================\n\n");

    /* -----------------------------------------------------------------
     * Configuration
     * ----------------------------------------------------------------- */
    const int n_vectors = 2048;       /* Number of KV cache vectors to process */
    const int d = TQ_HEAD_DIM;       /* Head dimension (128) */
    const size_t src_size = n_vectors * d * sizeof(float);
    const size_t dst_size = n_vectors * sizeof(block_tq3);

    printf("[Setup] Processing %d vectors of dimension %d\n", n_vectors, d);
    printf("         Input size:  %.2f KB (FP32)\n", src_size / 1024.0);
    printf("         Output size: %.2f KB (TQ3 quantized)\n", dst_size / 1024.0);
    printf("         Compression: %.1fx\n\n", (float)src_size / dst_size);

    /* -----------------------------------------------------------------
     * Step 1: Initialize CPU context to get rotation matrix
     * ----------------------------------------------------------------- */
    printf("[Step 1] Initializing TQ3 context...\n");
    tq_context ctx;
    int rc = tq_context_init(&ctx, 3, TQ_ROTATION_SEED);
    if (rc != 0) {
        fprintf(stderr, "%s Failed to initialize TQ3 context\n", FAIL);
        return EXIT_FAILURE;
    }
    printf("         %s Context initialized with %d levels\n", PASS, ctx.n_levels);

    /* -----------------------------------------------------------------
     * Step 2: Allocate host memory
     * ----------------------------------------------------------------- */
    printf("\n[Step 2] Allocating host memory...\n");

    float * h_src = (float *)malloc(src_size);
    float * h_dst = (float *)malloc(src_size);
    float * h_rotation = ctx.rotation;  /* Reuse context rotation matrix */
    float * h_rotation_bwd = ctx.rotation_bwd;  /* Reuse context rotation matrix */
    uint8_t * h_blocks = (uint8_t *)malloc(dst_size);

    printf("         %s Allocated %zu bytes host memory\n", PASS,
           src_size + dst_size);

    /* -----------------------------------------------------------------
     * Step 3: Generate test input vectors (random unit vectors)
     * ----------------------------------------------------------------- */
    printf("\n[Step 3] Generating test vectors...\n");

    for (int i = 0; i < n_vectors; i++) {
        float norm = 0.0f;
        for (int j = 0; j < d; j++) {
            h_src[i * d + j] = randn();
            norm += h_src[i * d + j] * h_src[i * d + j];
        }
        norm = sqrtf(norm);
        for (int j = 0; j < d; j++) {
            h_src[i * d + j] /= norm;  /* Normalize to unit vector */
        }
    }
    printf("         %s Generated %d random unit vectors\n", PASS, n_vectors);

    /* -----------------------------------------------------------------
     * Step 4: Allocate device memory
     * ----------------------------------------------------------------- */
    printf("\n[Step 4] Allocating device memory...\n");

    float * d_src = NULL;
    float * d_dst = NULL;
    float * d_rotation = NULL;
    float * d_rotation_bwd = NULL;
    uint8_t * d_blocks = NULL;

    CUDA_CHECK(cudaMalloc((void **)&d_src, src_size));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, src_size));
    CUDA_CHECK(cudaMalloc((void **)&d_rotation, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_rotation_bwd, d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_blocks, dst_size));

    printf("         %s Allocated device memory\n", PASS);

    /* -----------------------------------------------------------------
     * Step 5: Copy data to device
     * ----------------------------------------------------------------- */
    printf("\n[Step 5] Copying data to GPU...\n");

    CUDA_CHECK(cudaMemcpy(d_src, h_src, src_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation, h_rotation, d * d * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rotation_bwd, h_rotation_bwd, d * d * sizeof(float),
                          cudaMemcpyHostToDevice));

    printf("         %s Copied input vectors and rotation matrix\n", PASS);

    /* -----------------------------------------------------------------
     * Step 6: Initialize CUDA codebooks (constant memory)
     * ----------------------------------------------------------------- */
    printf("\n[Step 6] Initializing CUDA codebooks...\n");

    tq_cuda_init_codebooks();
    printf("         %s Codebooks loaded to constant memory\n", PASS);

    /* -----------------------------------------------------------------
     * Step 7: Create CUDA stream for async execution
     * ----------------------------------------------------------------- */
    printf("\n[Step 7] Creating CUDA stream...\n");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    printf("         %s Stream created\n", PASS);

    /* -----------------------------------------------------------------
     * Step 8: Call tq_cuda_quantize_tq3 kernel (multiple runs for timing)
     * ----------------------------------------------------------------- */
    printf("\n[Step 8] Running quantize kernel...\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int n_warmup = 3;
    const int n_runs = 20;
    float quant_times[n_runs];

    /* Warmup runs */
    for (int i = 0; i < n_warmup; i++) {
        tq_cuda_quantize_tq3(d_src, d_blocks, d_rotation, n_vectors, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /* Timed runs */
    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        // tq_cuda_quantize_tq3(d_src, d_blocks, d_rotation, n_vectors, stream);
        tq_cuda_quantize_tq3(d_src, d_blocks, d_rotation_bwd, n_vectors, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&quant_times[i], start, stop));
    }

    /* Compute average and min/max */
    float quant_min = quant_times[0], quant_max = quant_times[0], quant_sum = 0.0f;
    for (int i = 0; i < n_runs; i++) {
        quant_sum += quant_times[i];
        if (quant_times[i] < quant_min) quant_min = quant_times[i];
        if (quant_times[i] > quant_max) quant_max = quant_times[i];
    }
    float quant_avg = quant_sum / n_runs;

    printf("         %s Quantized %d vectors (%d runs)\n", PASS, n_vectors, n_runs);
    printf("         Average: %.3f ms (%.0f vec/s)\n", 
           quant_avg, n_vectors / (quant_avg / 1000.0));
    printf("         Min/Max: %.3f / %.3f ms\n", quant_min, quant_max);

    /* -----------------------------------------------------------------
     * Step 9: Copy quantized blocks back to host (for inspection)
     * ----------------------------------------------------------------- */
    printf("\n[Step 9] Copying quantized blocks to host...\n");

    CUDA_CHECK(cudaMemcpy(h_blocks, d_blocks, dst_size, cudaMemcpyDeviceToHost));
    printf("         %s Copied %zu bytes of quantized data\n", PASS, dst_size);

    /* Show first block as example */
    printf("\n         First quantized block:\n");
    block_tq3 * blk0 = (block_tq3 *)h_blocks;
    printf("           norm:   %.6f\n", blk0->norm);
    printf("           indices[0:8]: ");
    for (int i = 0; i < 8; i++) {
        printf("%02x ", blk0->indices[i]);
    }
    printf("\n");

    /* -----------------------------------------------------------------
     * Step 10: Call tq_cuda_dequantize_tq3 kernel (multiple runs for timing)
     * ----------------------------------------------------------------- */
    printf("\n[Step 10] Running dequantize kernel...\n");

    float dequant_times[n_runs];

    /* Warmup runs */
    for (int i = 0; i < n_warmup; i++) {
        tq_cuda_dequantize_tq3(d_blocks, d_dst, d_rotation, n_vectors, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    /* Timed runs */
    for (int i = 0; i < n_runs; i++) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        tq_cuda_dequantize_tq3(d_blocks, d_dst, d_rotation, n_vectors, stream);
        // tq_cuda_dequantize_tq3(d_blocks, d_dst, d_rotation_bwd, n_vectors, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&dequant_times[i], start, stop));
    }

    /* Compute average and min/max */
    float dequant_min = dequant_times[0], dequant_max = dequant_times[0], dequant_sum = 0.0f;
    for (int i = 0; i < n_runs; i++) {
        dequant_sum += dequant_times[i];
        if (dequant_times[i] < dequant_min) dequant_min = dequant_times[i];
        if (dequant_times[i] > dequant_max) dequant_max = dequant_times[i];
    }
    float dequant_avg = dequant_sum / n_runs;

    printf("         %s Dequantized %d vectors (%d runs)\n", PASS, n_vectors, n_runs);
    printf("         Average: %.3f ms (%.0f vec/s)\n", 
           dequant_avg, n_vectors / (dequant_avg / 1000.0));
    printf("         Min/Max: %.3f / %.3f ms\n", dequant_min, dequant_max);

    /* -----------------------------------------------------------------
     * Step 11: Copy dequantized results back to host
     * ----------------------------------------------------------------- */
    printf("\n[Step 11] Copying dequantized vectors to host...\n");

    CUDA_CHECK(cudaMemcpy(h_dst, d_dst, src_size, cudaMemcpyDeviceToHost));
    printf("         %s Copied dequantized vectors\n", PASS);

    /* -----------------------------------------------------------------
     * Step 12: Validate results (compute MSE)
     * ----------------------------------------------------------------- */
    printf("\n[Step 12] Validating results...\n");

    float total_mse = 0.0f;
    for (int i = 0; i < n_vectors; i++) {
        float mse = 0.0f;
        for (int j = 0; j < d; j++) {
            float diff = h_src[i * d + j] - h_dst[i * d + j];
            mse += diff * diff;
        }
        total_mse += mse;
    }
    float avg_mse = total_mse / n_vectors;

    printf("         Average MSE: %.6f (paper expects ~0.034 for TQ3)\n", 
           avg_mse);

    if (avg_mse < 0.1f) {
        printf("         %s MSE within acceptable range\n", PASS);
    } else {
        printf("         %s MSE higher than expected\n", FAIL);
    }

    /* -----------------------------------------------------------------
     * Cleanup
     * ----------------------------------------------------------------- */
    printf("\n[Cleanup] Freeing resources...\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_rotation));
    CUDA_CHECK(cudaFree(d_rotation_bwd));
    CUDA_CHECK(cudaFree(d_blocks));
    free(h_src);
    free(h_dst);
    free(h_blocks);

    printf("         %s All resources freed\n", PASS);

    /* -----------------------------------------------------------------
     * Summary
     * ----------------------------------------------------------------- */
    printf("\n=========================================================\n");
    printf("Summary (%d runs, %d vectors each):\n", n_runs, n_vectors);
    printf("  - Quantize:   %.3f ms avg (%.0f vec/s) [%.3f-%.3f]\n", 
           quant_avg, n_vectors / (quant_avg / 1000.0), quant_min, quant_max);
    printf("  - Dequantize: %.3f ms avg (%.0f vec/s) [%.3f-%.3f]\n", 
           dequant_avg, n_vectors / (dequant_avg / 1000.0), dequant_min, dequant_max);
    printf("  - MSE:        %.6f\n", avg_mse);
    printf("=========================================================\n");

    return EXIT_SUCCESS;
}
