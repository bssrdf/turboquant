"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate
=========================================================================
Python prototype implementation from the ICLR 2026 paper by Zandieh et al.
(arXiv:2504.19874)

Implements:
  - Algorithm 1: TurboQuant_mse  (MSE-optimal quantization)
  - Algorithm 2: TurboQuant_prod (Unbiased inner-product-optimal quantization)
  - QJL: 1-bit Quantized Johnson-Lindenstrauss transform

Built for Nexus Grove (ng-01) as a proof-of-concept prototype.
Not a CUDA kernel — this is NumPy reference code for validating the math
before integration into ik_llama.cpp.

Author: Jim / Claude collaboration
Date: 2026-03-25
"""

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.optimize import minimize_scalar
from typing import Tuple, Optional
import time
import json


# =============================================================================
# Section 1: Beta Distribution PDF for Hypersphere Coordinates (Lemma 1)
# =============================================================================

def beta_pdf(x: np.ndarray, d: int) -> np.ndarray:
    """
    PDF of a single coordinate of a uniformly random point on S^{d-1}.
    
    From Lemma 1:
      f_X(x) = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
    
    For x in [-1, 1].
    In high dimensions this converges to N(0, 1/d).
    """
    if d <= 2:
        raise ValueError("Dimension d must be >= 3 for this distribution")
    
    coeff = gamma_fn(d / 2.0) / (np.sqrt(np.pi) * gamma_fn((d - 1) / 2.0))
    # Clip to avoid numerical issues at boundaries
    x_clipped = np.clip(x, -1 + 1e-15, 1 - 1e-15)
    return coeff * np.power(1.0 - x_clipped**2, (d - 3) / 2.0)


# =============================================================================
# Section 2: Lloyd-Max Optimal Scalar Quantizer (Eq. 4)
# =============================================================================

def compute_lloyd_max_codebook(d: int, b: int, max_iter: int = 200, 
                                tol: float = 1e-12) -> np.ndarray:
    """
    Compute optimal Lloyd-Max codebook for the Beta distribution on [-1, 1]
    induced by random rotation of unit vectors in R^d.
    
    This solves the continuous 1D k-means problem from Eq. (4):
      min_{c_1,...,c_{2^b}} sum_i integral |x - c_i|^2 * f_X(x) dx
    
    Uses iterative Lloyd-Max algorithm:
      1. Given centroids, compute optimal boundaries (midpoints)
      2. Given boundaries, compute optimal centroids (conditional means)
      3. Repeat until convergence
    
    Args:
        d: Vector dimension
        b: Bit-width (number of bits per coordinate)
        max_iter: Maximum Lloyd-Max iterations
        tol: Convergence tolerance on centroid movement
    
    Returns:
        Sorted array of 2^b centroid values
    """
    n_levels = 2**b
    
    # For high d, the distribution is approximately N(0, 1/d).
    # Initialize centroids uniformly in the range where most mass lives.
    sigma = 1.0 / np.sqrt(d)
    # Span ~3 sigma on each side
    centroids = np.linspace(-3 * sigma, 3 * sigma, n_levels)
    
    # Numerical integration grid — fine enough for good codebook quality
    n_grid = 10000
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, n_grid)
    dx = x_grid[1] - x_grid[0]
    pdf_vals = beta_pdf(x_grid, d)
    
    for iteration in range(max_iter):
        # Step 1: Compute boundaries (midpoints between consecutive centroids)
        boundaries = np.concatenate([
            [-1.0],
            0.5 * (centroids[:-1] + centroids[1:]),
            [1.0]
        ])
        
        # Step 2: Compute optimal centroids as conditional means within each bin
        new_centroids = np.zeros(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (x_grid >= lo) & (x_grid < hi)
            if not np.any(mask):
                # Empty bin — keep old centroid
                new_centroids[i] = centroids[i]
                continue
            
            weighted_x = np.sum(x_grid[mask] * pdf_vals[mask]) * dx
            total_weight = np.sum(pdf_vals[mask]) * dx
            
            if total_weight < 1e-20:
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = weighted_x / total_weight
        
        # Check convergence
        shift = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        
        if shift < tol:
            break
    
    return np.sort(centroids)


def compute_codebook_mse(centroids: np.ndarray, d: int) -> float:
    """
    Compute the MSE cost C(f_X, b) for a given codebook.
    This is d * C(f_X, b) = expected ||x - x_hat||^2 per Theorem 1.
    """
    n_grid = 10000
    x_grid = np.linspace(-1 + 1e-10, 1 - 1e-10, n_grid)
    dx = x_grid[1] - x_grid[0]
    pdf_vals = beta_pdf(x_grid, d)
    
    # For each grid point, find nearest centroid
    # Shape: (n_grid, n_levels) -> distances
    dists = np.abs(x_grid[:, None] - centroids[None, :])
    nearest_idx = np.argmin(dists, axis=1)
    nearest_centroid = centroids[nearest_idx]
    
    # MSE per coordinate
    mse_per_coord = np.sum((x_grid - nearest_centroid)**2 * pdf_vals) * dx
    
    # Total MSE for d-dimensional vector
    return d * mse_per_coord


# =============================================================================
# Section 3: Random Rotation Matrix Generation
# =============================================================================

def generate_random_rotation(d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random orthogonal rotation matrix via QR decomposition
    of a random Gaussian matrix.
    
    This is the Π matrix from Algorithm 1, line 2.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d))
    Q, R = np.linalg.qr(G)
    # Ensure proper rotation (det = +1) by fixing sign ambiguity
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    Q = Q * signs[None, :]
    return Q


# =============================================================================
# Section 4: QJL — Quantized Johnson-Lindenstrauss (Definition 1)
# =============================================================================

class QJL:
    """
    1-bit Quantized Johnson-Lindenstrauss transform.
    
    From Definition 1:
      Q_qjl(x) = sign(S · x)
      Q_qjl^{-1}(z) = sqrt(π/2) / d · S^T · z
    
    Provides unbiased inner product estimates with zero memory overhead.
    """
    
    def __init__(self, d: int, seed: Optional[int] = None):
        self.d = d
        rng = np.random.default_rng(seed)
        self.S = rng.standard_normal((d, d))
    
    def quantize(self, x: np.ndarray) -> np.ndarray:
        """Quantize to sign bits: sign(S · x)"""
        projected = self.S @ x
        return np.sign(projected).astype(np.int8)
    
    def dequantize(self, z: np.ndarray, residual_norm: float) -> np.ndarray:
        """
        Dequantize: sqrt(π/2) / d · γ · S^T · z
        where γ = ||residual||_2
        """
        scale = np.sqrt(np.pi / 2.0) / self.d * residual_norm
        return scale * (self.S.T @ z.astype(np.float64))


# =============================================================================
# Section 5: TurboQuant_mse — Algorithm 1
# =============================================================================

class TurboQuantMSE:
    """
    MSE-optimal TurboQuant (Algorithm 1).
    
    Quantization:
      1. Rotate: y = Π · x
      2. For each coordinate y_j, find nearest centroid index
      
    Dequantization:
      1. Replace indices with centroid values -> y_hat
      2. Rotate back: x_hat = Π^T · y_hat
    """
    
    def __init__(self, d: int, b: int, rotation_seed: Optional[int] = None,
                 codebook: Optional[np.ndarray] = None):
        """
        Args:
            d: Vector dimension
            b: Bit-width per coordinate
            rotation_seed: Seed for reproducible rotation matrix
            codebook: Pre-computed codebook (if None, computes via Lloyd-Max)
        """
        self.d = d
        self.b = b
        self.n_levels = 2**b
        
        # Line 2: Generate random rotation matrix
        self.Pi = generate_random_rotation(d, seed=rotation_seed)
        
        # Line 3: Construct codebook via Lloyd-Max
        if codebook is not None:
            self.codebook = np.sort(codebook)
        else:
            print(f"  Computing Lloyd-Max codebook (d={d}, b={b})...")
            t0 = time.time()
            self.codebook = compute_lloyd_max_codebook(d, b)
            print(f"  Codebook computed in {time.time()-t0:.3f}s: {self.codebook}")
    
    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Algorithm 1, Lines 4-7: Quantize a vector.
        
        Args:
            x: Input vector of shape (d,). Need not be unit norm —
               we store the norm separately.
        
        Returns:
            (indices, norm): Quantized index array and original L2 norm
        """
        norm = np.linalg.norm(x)
        if norm < 1e-15:
            return np.zeros(self.d, dtype=np.int32), 0.0
        
        # Normalize to unit sphere
        x_unit = x / norm
        
        # Line 5: y = Π · x
        y = self.Pi @ x_unit
        
        # Line 6: Find nearest centroid for each coordinate
        # Shape: (d, n_levels) -> pick argmin per coordinate
        dists = np.abs(y[:, None] - self.codebook[None, :])
        indices = np.argmin(dists, axis=1).astype(np.int32)
        
        return indices, norm
    
    def dequantize(self, indices: np.ndarray, norm: float) -> np.ndarray:
        """
        Algorithm 1, Lines 8-11: Dequantize.
        
        Args:
            indices: Index array from quantize()
            norm: Original L2 norm
        
        Returns:
            Reconstructed vector of shape (d,)
        """
        # Line 9: y_hat_j = c_{idx_j}
        y_hat = self.codebook[indices]
        
        # Line 10: x_hat = Π^T · y_hat
        x_hat = self.Pi.T @ y_hat
        
        # Rescale by original norm
        return x_hat * norm
    
    def compress_size_bits(self) -> int:
        """Total bits used to store one quantized vector (excluding norm)."""
        return self.d * self.b


# =============================================================================
# Section 6: TurboQuant_prod — Algorithm 2
# =============================================================================

class TurboQuantProd:
    """
    Inner-product-optimal TurboQuant (Algorithm 2).
    
    Two-stage approach:
      Stage 1: Apply TurboQuant_mse at (b-1) bits
      Stage 2: Apply QJL on the residual (1 bit per coordinate)
    
    This eliminates the inner product bias that MSE-optimal quantizers have.
    Total bit-width: b = (b-1) + 1
    """
    
    def __init__(self, d: int, b: int, rotation_seed: Optional[int] = None,
                 qjl_seed: Optional[int] = None,
                 codebook: Optional[np.ndarray] = None):
        """
        Args:
            d: Vector dimension
            b: Total bit-width per coordinate (must be >= 2)
            rotation_seed: Seed for MSE quantizer rotation
            qjl_seed: Seed for QJL projection matrix
            codebook: Pre-computed codebook for the (b-1) MSE stage
        """
        if b < 2:
            raise ValueError("TurboQuant_prod requires b >= 2 (1 bit for MSE + 1 for QJL)")
        
        self.d = d
        self.b = b
        
        # Line 2: Instantiate TurboQuant_mse with bit-width (b-1)
        self.mse_quant = TurboQuantMSE(d, b - 1, rotation_seed=rotation_seed,
                                        codebook=codebook)
        
        # Line 3: Generate QJL random projection matrix
        self.qjl = QJL(d, seed=qjl_seed)
    
    def quantize(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Algorithm 2, Lines 4-8: Quantize a vector.
        
        Returns:
            (mse_indices, qjl_signs, residual_norm, original_norm)
        """
        original_norm = np.linalg.norm(x)
        if original_norm < 1e-15:
            return (np.zeros(self.d, dtype=np.int32),
                    np.zeros(self.d, dtype=np.int8),
                    0.0, 0.0)
        
        # Line 5: MSE quantize
        mse_indices, norm = self.mse_quant.quantize(x)
        
        # Line 6: Compute residual r = x - DeQuant_mse(idx)
        x_mse_reconstructed = self.mse_quant.dequantize(mse_indices, norm)
        residual = x - x_mse_reconstructed
        residual_norm = np.linalg.norm(residual)
        
        # Line 7: QJL on residual
        if residual_norm < 1e-15:
            qjl_signs = np.zeros(self.d, dtype=np.int8)
        else:
            # Normalize residual before QJL (QJL expects unit vectors)
            qjl_signs = self.qjl.quantize(residual / residual_norm)
        
        # Line 8: output (idx, qjl, ||r||_2)
        return mse_indices, qjl_signs, residual_norm, original_norm
    
    def dequantize(self, mse_indices: np.ndarray, qjl_signs: np.ndarray,
                   residual_norm: float, original_norm: float) -> np.ndarray:
        """
        Algorithm 2, Lines 9-12: Dequantize.
        
        Returns:
            Reconstructed vector of shape (d,)
        """
        # Line 10: x_hat_mse = DeQuant_mse(idx)
        x_mse = self.mse_quant.dequantize(mse_indices, original_norm)
        
        # Line 11: x_hat_qjl = sqrt(π/2)/d · γ · S^T · qjl
        x_qjl = self.qjl.dequantize(qjl_signs, residual_norm)
        
        # Line 12: output x_hat_mse + x_hat_qjl
        return x_mse + x_qjl
    
    def compress_size_bits(self) -> int:
        """Total bits per vector: (b-1)*d for MSE + d for QJL + 32 for norm."""
        return self.d * self.b + 32  # +32 for the residual norm float


# =============================================================================
# Section 7: Compression Ratio Calculator
# =============================================================================

def compression_report(d: int, b: int, original_bits: int = 16) -> dict:
    """
    Calculate compression ratio for KV cache quantization.
    
    Args:
        d: Head dimension (e.g., 128 for most modern transformers)
        b: TurboQuant bit-width
        original_bits: Original precision (16 for fp16/bf16)
    
    Returns:
        Dict with compression metrics
    """
    original_size = d * original_bits
    turbo_size = d * b + 32  # +32 bits for norm storage
    ratio = original_size / turbo_size
    
    return {
        "dimension": d,
        "bit_width": b,
        "original_bits_per_vector": original_size,
        "turboquant_bits_per_vector": turbo_size,
        "compression_ratio": f"{ratio:.2f}x",
        "memory_fraction": f"{turbo_size/original_size:.1%}",
    }


# =============================================================================
# Section 8: Test Suite
# =============================================================================

def run_tests():
    """
    Validate TurboQuant against the paper's theoretical predictions.
    """
    print("=" * 70)
    print("TurboQuant Prototype — Validation Suite")
    print("Reference: Zandieh et al., ICLR 2026 (arXiv:2504.19874)")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Test 1: Codebook quality — compare MSE to paper's Table (Theorem 1)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Lloyd-Max Codebook Quality vs Paper's Theorem 1")
    print("-" * 50)
    
    # Paper's expected MSE values for b=1,2,3,4 (from Theorem 1):
    expected_mse = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}
    
    d_test = 128  # Typical transformer head dimension
    
    codebooks = {}
    for b in [1, 2, 3, 4]:
        cb = compute_lloyd_max_codebook(d_test, b)
        mse = compute_codebook_mse(cb, d_test)
        codebooks[b] = cb
        paper_val = expected_mse[b]
        ratio = mse / paper_val
        status = "✓" if 0.5 < ratio < 2.0 else "✗"
        print(f"  b={b}: MSE={mse:.6f}  (paper≈{paper_val})  ratio={ratio:.3f}  {status}")
    
    # -------------------------------------------------------------------------
    # Test 2: TurboQuant_mse round-trip
    # -------------------------------------------------------------------------
    print("\n[Test 2] TurboQuant_mse Round-trip (d=128)")
    print("-" * 50)
    
    rng = np.random.default_rng(42)
    n_vectors = 1000
    
    for b in [2, 3, 4]:
        quant = TurboQuantMSE(d_test, b, rotation_seed=42, codebook=codebooks[b])
        
        total_mse = 0.0
        for _ in range(n_vectors):
            x = rng.standard_normal(d_test)
            x = x / np.linalg.norm(x)  # Unit vector
            
            idx, norm = quant.quantize(x)
            x_hat = quant.dequantize(idx, norm)
            
            total_mse += np.sum((x - x_hat)**2)
        
        avg_mse = total_mse / n_vectors
        paper_val = expected_mse[b]
        print(f"  b={b}: Avg MSE={avg_mse:.6f}  (paper≈{paper_val})")
    
    # -------------------------------------------------------------------------
    # Test 3: TurboQuant_prod unbiasedness
    # -------------------------------------------------------------------------
    print("\n[Test 3] TurboQuant_prod Inner Product Unbiasedness (d=128, b=3)")
    print("-" * 50)
    
    b = 3
    quant_prod = TurboQuantProd(d_test, b, rotation_seed=42, qjl_seed=99,
                                 codebook=codebooks[b - 1])
    
    # Test: E[<y, x_hat>] should equal <y, x>
    n_trials = 500
    bias_samples = []
    
    for _ in range(n_trials):
        x = rng.standard_normal(d_test)
        x = x / np.linalg.norm(x)
        y = rng.standard_normal(d_test)
        
        true_ip = np.dot(y, x)
        
        # Average over multiple quantization rounds (randomness in rotation/QJL)
        # For a single instance, the rotation is fixed, so we just measure once
        idx, qjl_signs, r_norm, o_norm = quant_prod.quantize(x)
        x_hat = quant_prod.dequantize(idx, qjl_signs, r_norm, o_norm)
        est_ip = np.dot(y, x_hat)
        
        bias_samples.append(est_ip - true_ip)
    
    mean_bias = np.mean(bias_samples)
    std_bias = np.std(bias_samples)
    print(f"  Mean bias: {mean_bias:.6f}  (should be ≈0)")
    print(f"  Std of error: {std_bias:.6f}")
    print(f"  |Mean bias| / Std: {abs(mean_bias)/std_bias:.4f}  (should be small)")
    
    # -------------------------------------------------------------------------
    # Test 4: Compression ratios for ng-01 relevant scenarios
    # -------------------------------------------------------------------------
    print("\n[Test 4] Compression Ratios — KV Cache Scenarios")
    print("-" * 50)
    
    # Qwen3.5-27B: head_dim=128, typical for modern transformers
    # 70B models: also typically head_dim=128
    for b in [2, 3, 4]:
        report = compression_report(d=128, b=b, original_bits=16)
        print(f"  b={b}: {report['compression_ratio']} compression "
              f"({report['memory_fraction']} of original)")
    
    # -------------------------------------------------------------------------
    # Test 5: Quantization speed benchmark
    # -------------------------------------------------------------------------
    print("\n[Test 5] Quantization Speed (d=128, n=10000 vectors)")
    print("-" * 50)
    
    quant_mse = TurboQuantMSE(d_test, 3, rotation_seed=42, codebook=codebooks[3])
    vectors = rng.standard_normal((10000, d_test))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # MSE quantize
    t0 = time.time()
    for i in range(len(vectors)):
        quant_mse.quantize(vectors[i])
    t_mse = time.time() - t0
    print(f"  TurboQuant_mse (b=3): {t_mse:.3f}s  "
          f"({len(vectors)/t_mse:.0f} vectors/sec)")
    
    # -------------------------------------------------------------------------
    # Test 6: Simulated KV cache compression for Qwen3.5-27B
    # -------------------------------------------------------------------------
    print("\n[Test 6] Simulated KV Cache — Qwen3.5-27B Dense")
    print("-" * 50)
    
    # Qwen3.5-27B approximate KV cache params:
    #   - num_layers: ~32 (estimated for 27B dense)
    #   - num_kv_heads: ~8 (GQA typical)  
    #   - head_dim: 128
    #   - For 8K context: 8192 tokens
    
    n_layers = 32
    n_kv_heads = 8
    head_dim = 128
    context_lengths = [4096, 8192, 16384, 32768]
    
    print(f"  Model: Qwen3.5-27B Dense (est. {n_layers}L, {n_kv_heads} KV heads, "
          f"d={head_dim})")
    print(f"  KV cache = 2 (K+V) × layers × kv_heads × seq_len × head_dim × precision")
    print()
    
    for ctx_len in context_lengths:
        # FP16 baseline
        fp16_bytes = 2 * n_layers * n_kv_heads * ctx_len * head_dim * 2  # 2 bytes per fp16
        fp16_gb = fp16_bytes / (1024**3)
        
        # TurboQuant at 3.5 bits (paper's quality-neutral setting)
        tq_bits_per_val = 3.5
        tq_bytes = 2 * n_layers * n_kv_heads * ctx_len * head_dim * tq_bits_per_val / 8
        tq_gb = tq_bytes / (1024**3)
        
        ratio = fp16_gb / tq_gb
        print(f"  {ctx_len:>6} tokens: FP16={fp16_gb:.2f} GB → TQ@3.5b={tq_gb:.2f} GB "
              f"({ratio:.1f}x compression)")
    
    # -------------------------------------------------------------------------
    # Test 7: 70B model projection across 3x RTX 3090
    # -------------------------------------------------------------------------
    print("\n[Test 7] Projected KV Cache — 70B Model on 3× RTX 3090 (72GB)")
    print("-" * 50)
    
    # 70B model approximate params (Llama-3.1-70B style):
    n_layers_70b = 80
    n_kv_heads_70b = 8   # GQA
    head_dim_70b = 128
    
    # Weight memory estimate: 70B params at Q4 ≈ ~35-40GB
    weight_gb = 38.0  # Conservative Q4 estimate
    total_vram = 72.0
    available_for_kv = total_vram - weight_gb
    
    print(f"  Model: 70B (Q4_K_M ≈ {weight_gb:.0f} GB weights)")
    print(f"  Total VRAM: {total_vram:.0f} GB")
    print(f"  Available for KV cache: {available_for_kv:.0f} GB")
    print()
    
    for tq_bits in [16.0, 3.5, 2.5]:
        bytes_per_token = (2 * n_layers_70b * n_kv_heads_70b * head_dim_70b 
                          * tq_bits / 8)
        gb_per_token = bytes_per_token / (1024**3)
        max_tokens = int(available_for_kv / gb_per_token)
        
        label = "FP16" if tq_bits == 16.0 else f"TQ@{tq_bits}b"
        print(f"  {label:>8}: {bytes_per_token:.0f} bytes/token → "
              f"~{max_tokens:,} tokens max context")
    
    print()
    print("=" * 70)
    print("All tests complete.")
    print("=" * 70)


# =============================================================================
# Section 9: Codebook Export (for future C/CUDA integration)
# =============================================================================

def export_codebooks(d: int, bit_widths: list = [1, 2, 3, 4],
                     output_path: str = "turboquant_codebooks.json") -> dict:
    """
    Precompute and export Lloyd-Max codebooks for use in C/CUDA implementations.
    
    The codebooks only depend on (d, b), so they can be computed once and
    embedded as constants in ik_llama.cpp.
    """
    result = {"dimension": d, "codebooks": {}}
    
    for b in bit_widths:
        print(f"Computing codebook for d={d}, b={b}...")
        cb = compute_lloyd_max_codebook(d, b)
        mse = compute_codebook_mse(cb, d)
        result["codebooks"][str(b)] = {
            "bit_width": b,
            "n_levels": 2**b,
            "centroids": cb.tolist(),
            "expected_mse": float(mse),
        }
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nCodebooks exported to: {output_path}")
    return result


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    run_tests()
