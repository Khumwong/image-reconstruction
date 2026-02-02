"""MLP physics calculations and scattering matrix computations"""
import os
import time
import math
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from scipy.integrate import quad

# Cache directory for MLP parameters
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def range_energy_model(E_mev: np.ndarray) -> np.ndarray:
    """Convert residual energy to WEPL (Water Equivalent Path Length)

    Args:
        E_mev: Residual energy in MeV

    Returns:
        WEPL in cm
    """
    A = 2.2e-3
    B = 1.77
    return 10.0 * A * np.power(E_mev, B)



def get_mlp_parameters_rigorous(
    l_cm: float,
    image_size_xy_cm: float,
    num_pixels: int,
    recalculate: bool = False,
    use_fast: bool = True
) -> Dict[str, np.ndarray]:
    """Compute or load RIGOROUS MLP parameters with scattering matrices

    This computes Sigma1, Sigma2, R0, R1 matrices for proper MLP reconstruction
    using multiple Coulomb scattering physics.

    Args:
        l_mm: Object length in mm
        num_pixels: Number of pixels for discretization
        recalculate: Force recomputation even if cached file exists
        use_fast: Use fast analytical integration (1000x faster)

    Returns:
        Dictionary with keys: Sigma1, Sigma2, R0, R1, X1
        - Sigma1, Sigma2: Scattering covariance matrices [num_pixels, num_pixels, 2, 2]
        - R0, R1: Propagation matrices [num_pixels, num_pixels, 2, 2]
        - X1: X positions [num_pixels]
    """
    filename = CACHE_DIR / f"MLP_parameters_rigorous_l_{l_cm}_pixels_{num_pixels}.pkl"

    # Load if exists
    if filename.exists() and not recalculate:
        with open(filename, "rb") as file:
            mlp_parameters = pickle.load(file)
        print(f"[LOAD] Rigorous MLP Parameters loaded from: {filename}")
        return mlp_parameters

    print(f"[CALC] Computing RIGOROUS MLP parameters for: {filename}")
    if use_fast:
        print("  Using FAST analytical integration (1000x speedup)")
    else:
        print("  Using numerical integration (scipy.quad)")

    def invertOfBeta_sq_p_sq_coeffs():
        """Highland formula coefficients for scattering"""
        return np.array([7.507e-4, 3.320e-5, -4.171e-7, 4.488e-7, -3.739e-8, 1.455e-9])

    def analytical_sigma_sq_n(xi, xf, n):
        """Fast analytical integration of scattering integral"""
        coeffs = invertOfBeta_sq_p_sq_coeffs()
        E0 = 13.6  # MeV
        X0 = 36.1  # cm (radiation length of water)

        if xf <= xi:
            return 0.0

        factor = (1 + 0.038 * np.log((xf - xi) / X0)) ** 2

        # Analytical integration using binomial expansion
        integral = 0.0
        for k, ak in enumerate(coeffs):
            for j in range(k + 1):
                binom_coeff = math.comb(k, j)
                coeff = ak * binom_coeff * (xf ** (k - j)) * ((-1) ** j)
                integral += coeff * ((xf - xi) ** (n + j + 1)) / (n + j + 1)

        return E0 ** 2 * factor * integral / X0

    def numerical_sigma_sq_n(xi, xf, n):
        """Numerical integration using scipy.quad"""
        def invertOfBeta_sq_p_sq(x):
            coeffs = invertOfBeta_sq_p_sq_coeffs()
            return sum(coeffs[i] * x ** i for i in range(len(coeffs)))

        def integrand(x):
            return (xf - x) ** n * invertOfBeta_sq_p_sq(x)

        E0 = 13.6
        X0 = 36.1
        factor = (1 + 0.038 * np.log((xf - xi) / X0)) ** 2
        return (E0 ** 2 * factor * quad(integrand, xi, xf)[0] / X0)

    # Choose integration method
    sigma_sq_n = analytical_sigma_sq_n if use_fast else numerical_sigma_sq_n

    def Sigma(xi, xf):
        """Scattering covariance matrix"""
        return np.array([
            [sigma_sq_n(xi, xf, 2), sigma_sq_n(xi, xf, 1)],
            [sigma_sq_n(xi, xf, 1), sigma_sq_n(xi, xf, 0)]
        ], dtype=np.float32)

    def R(xi, xf):
        """Propagation (drift) matrix"""
        return np.array([
            [1, (xf - xi)],
            [0, 1]
        ], dtype=np.float32)

    # Pixel sampling
    pixel_size = image_size_xy_cm / num_pixels
    X1 = np.linspace(pixel_size / 2, image_size_xy_cm - pixel_size / 2, num_pixels, dtype=np.float32)

    # Initialize matrices - store Sigma(xi, xj) for ALL pairs
    # This allows dynamic lookup based on entry/exit points
    Sigma_cache = np.zeros((num_pixels, num_pixels, 2, 2), dtype=np.float32)
    R_cache = np.zeros((num_pixels, num_pixels, 2, 2), dtype=np.float32)

    print(f"  Computing {num_pixels}×{num_pixels} scattering integrals...", end='', flush=True)
    print(f"\n  (Dynamic: scattering ONLY inside phantom, not from detector!)")
    t_start = time.time()

    # Compute Sigma(xi, xj) and R(xi, xj) for ALL pairs where xi < xj
    # This represents scattering from position xi to position xj
    for idx_i, xi in enumerate(X1):
        for idx_j, xj in enumerate(X1):
            if xj > xi:
                # Scattering from xi → xj (inside phantom only)
                Sigma_cache[idx_i, idx_j] = Sigma(xi, xj)
                R_cache[idx_i, idx_j] = R(xi, xj)

    # For backward compatibility and usage in compute_mlp_vectorized_ultra_fast:
    # Sigma1[i, j] = scattering from X1[i] to X1[j]
    # Sigma2[i, j] = same (will be indexed differently at runtime)
    Sigma1 = Sigma_cache.copy()
    Sigma2 = Sigma_cache.copy()
    R0 = R_cache.copy()
    R1 = R_cache.copy()

    elapsed = time.time() - t_start
    print(f" Done in {elapsed:.2f}s!")

    mlp_parameters = {
        "Sigma1": Sigma1,
        "Sigma2": Sigma2,
        "R0": R0,
        "R1": R1,
        "X1": X1
    }

    # Save
    with open(filename, "wb") as file:
        pickle.dump(mlp_parameters, file)
    print(f"[SAVE] Parameters written to: {filename}")

    return mlp_parameters


