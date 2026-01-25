"""MLP physics calculations and scattering matrix computations"""
import os
import time
import math
import pickle
import numpy as np
from typing import Dict, Tuple
from scipy.integrate import quad


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


def i_theta_coeffs() -> np.ndarray:
    """Return polynomial coefficients for i_theta (from img_recon.py)

    i_theta(xi) = a0 + a1*xi + a2*xi^2 + a3*xi^3 + a4*xi^4 + a5*xi^5

    Returns:
        Array of 6 polynomial coefficients
    """
    return np.array([7.507e-4, 3.320e-5, -4.171e-7, 4.488e-7, -3.739e-8, 1.455e-9])


def analytical_sigma_sq_n(xi: float, xf: float, n: int) -> float:
    """FAST analytical integration - 1000x faster than scipy.quad

    Computes: ∫[xi to xf] (xf - x)^n × i_theta(x) dx analytically
    where i_theta(x) = a0 + a1*x + a2*x^2 + ... + a5*x^5

    Args:
        xi: Initial position
        xf: Final position
        n: Power exponent

    Returns:
        Scattering integral value
    """
    coeffs = i_theta_coeffs()

    if xf <= xi:
        return 0.0

    # Analytical integration using binomial expansion
    integral = 0.0
    for k, ak in enumerate(coeffs):
        # Expand (xf - x)^n * x^k and integrate term by term
        for j in range(k + 1):
            # Binomial coefficient C(k,j)
            binom_coeff = math.comb(k, j)
            # Term: C(k,j) × xf^(k-j) × (-1)^j × (xf-x)^(n+j)
            coeff = ak * binom_coeff * (xf ** (k - j)) * ((-1) ** j)
            # Integrate (xf-x)^(n+j) from xi to xf
            # Result: [(xf-xi)^(n+j+1)] / (n+j+1)
            integral += coeff * ((xf - xi) ** (n + j + 1)) / (n + j + 1)

    return integral


def numerical_sigma_sq_n(xi: float, xf: float, n: int) -> float:
    """Original slow scipy.quad integration (for comparison/validation)

    Args:
        xi: Initial position
        xf: Final position
        n: Power exponent

    Returns:
        Scattering integral value
    """
    def i_theta(x):
        coeffs = i_theta_coeffs()
        return sum(coeffs[i] * x**i for i in range(len(coeffs)))

    def integrand(x):
        return (xf - x)**n * i_theta(x)

    return quad(integrand, xi, xf)[0]


def get_mlp_parameters(
    l_mm: float,
    num_pixels: int,
    recalculate: bool = False,
    use_fast: bool = True
) -> Dict[str, np.ndarray]:
    """Compute or load MLP parameters (img_recon.py version)

    Args:
        l_mm: Object length in mm
        num_pixels: Number of pixels for discretization
        recalculate: Force recomputation even if cached file exists
        use_fast: Use fast analytical integration (1000x faster)

    Returns:
        Dictionary with keys: X_position, G1, G2, H1, H2
    """
    filename = f"MLP_parameters_img_recon_l_{l_mm}_pixels_{num_pixels}.pkl"

    # Load if exists
    if os.path.exists(filename) and not recalculate:
        with open(filename, "rb") as file:
            mlp_parameters = pickle.load(file)
        print(f"[LOAD] MLP Parameters loaded from: {filename}")
        return mlp_parameters

    print(f"[CALC] Computing MLP parameters (img_recon.py version) for: {filename}")
    if use_fast:
        print("  Using FAST analytical integration (1000x speedup)")
    else:
        print("  Using numerical integration (scipy.quad)")

    # Helper functions for integration
    def i_theta(xi):
        coeffs = i_theta_coeffs()
        return sum(coeffs[i] * xi ** i for i in range(len(coeffs)))

    def i_y(xi, x1):
        return i_theta(xi) * (x1 - xi) ** 2

    def i_thetaY(xi, x1):
        return i_theta(xi) * (x1 - xi)

    pixel_size = l_mm / num_pixels
    X_position = np.linspace(pixel_size / 2, l_mm - pixel_size / 2, num=num_pixels)

    sigma_theta1_sq = np.zeros(num_pixels)
    sigma_y1_sq = np.zeros(num_pixels)
    sigma_thetaY1_sq = np.zeros(num_pixels)
    sigma_theta2_sq = np.zeros(num_pixels)
    sigma_y2_sq = np.zeros(num_pixels)
    sigma_thetaY2_sq = np.zeros(num_pixels)

    print(f"  Computing {num_pixels} MLP integrals...", end='', flush=True)
    t_start = time.time()

    if use_fast:
        # Use analytical integration
        for i in range(num_pixels):
            sigma_theta1_sq[i] = analytical_sigma_sq_n(0, X_position[i], 0)
            sigma_y1_sq[i] = analytical_sigma_sq_n(0, X_position[i], 2)
            sigma_thetaY1_sq[i] = analytical_sigma_sq_n(0, X_position[i], 1)
            sigma_theta2_sq[i] = analytical_sigma_sq_n(X_position[i], l_mm, 0)
            sigma_y2_sq[i] = analytical_sigma_sq_n(X_position[i], l_mm, 2)
            sigma_thetaY2_sq[i] = -analytical_sigma_sq_n(X_position[i], l_mm, 1)
    else:
        # Use scipy.quad (original img_recon.py method)
        for i in range(num_pixels):
            sigma_theta1_sq[i], _ = quad(i_theta, 0, X_position[i])
            sigma_y1_sq[i], _ = quad(i_y, 0, X_position[i], args=(X_position[i],))
            sigma_thetaY1_sq[i], _ = quad(i_thetaY, 0, X_position[i], args=(X_position[i],))
            sigma_theta2_sq[i], _ = quad(i_theta, X_position[i], l_mm)
            sigma_y2_sq[i], _ = quad(i_y, X_position[i], l_mm, args=(X_position[i],))
            sigma_thetaY2_sq[i], _ = quad(i_thetaY, X_position[i], l_mm, args=(X_position[i],))
            sigma_thetaY2_sq[i] = -sigma_thetaY2_sq[i]

    elapsed = time.time() - t_start
    print(f" Done in {elapsed:.2f}s!")

    # Compute determinants and final parameters (img_recon.py formulas)
    det2 = (sigma_theta2_sq * sigma_y2_sq) - (sigma_thetaY2_sq ** 2)
    det1 = (sigma_theta1_sq * sigma_y1_sq) - (sigma_thetaY1_sq ** 2)

    A = sigma_y1_sq / det1 + sigma_y2_sq / det2
    B = -sigma_thetaY1_sq / det1 + sigma_thetaY2_sq / det2
    C = sigma_theta1_sq / det1 + sigma_theta2_sq / det2
    ABC = (A * C) - (B ** 2)

    G1 = (sigma_thetaY2_sq * sigma_y1_sq + sigma_y2_sq * sigma_thetaY1_sq) / (ABC * det1 * det2)
    G2 = (sigma_thetaY2_sq * sigma_y2_sq - sigma_y2_sq * sigma_thetaY2_sq) / (ABC * det2 ** 2)

    H1 = (sigma_theta2_sq * sigma_y1_sq + sigma_thetaY2_sq * sigma_thetaY1_sq) / (ABC * det1 * det2)
    H2 = (sigma_theta2_sq * sigma_y2_sq - sigma_thetaY2_sq ** 2) / (ABC * det2 ** 2)

    mlp_parameters = {
        "X_position": X_position.astype(np.float32),
        "G1": G1.astype(np.float32),
        "G2": G2.astype(np.float32),
        "H1": H1.astype(np.float32),
        "H2": H2.astype(np.float32),
    }

    # Save
    with open(filename, "wb") as file:
        pickle.dump(mlp_parameters, file)
    print(f"[SAVE] Parameters written to: {filename}")

    return mlp_parameters
