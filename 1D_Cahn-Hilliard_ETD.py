import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

np.random.seed(42)  # For reproducibility
N = 100  # Number of grid points
L = 100  # Length of the domain
h = L / N  # Step size
M = 1.0  # Mobility constant
kappa = 1.0  # Gradient energy coefficient
x = np.linspace(0, L, N, endpoint=False)
# Initialize randomly probability field phi as a 1D array
phi = np.random.uniform(-1, 1, size=N)
# Wavenumbers
k = 2 * np.pi * fftfreq(N, d=h)
k2 = k**2
k4 = k2**2

# Time parameters
dt = 0.1      # Time step
n_steps = 10000 # Number of time steps
save_every = 50


# Precompute ETD2 coefficients
L_hat = -k4
E = np.exp(L_hat * dt)
E2 = np.exp(L_hat * dt / 2)

# ETD2 denominators (avoid division by 0)
L_hat[L_hat == 0] = 1e-20
inv_L_hat = 1 / L_hat
phi_func = (E - 1) * inv_L_hat
phi_func2 = (E2 - 1) * inv_L_hat

# Time loop
for step in range(n_steps):

    # First half-step: compute nonlinear term
    phi_hat = fft(phi)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = ifft(lap_phi_hat).real
    mu = phi**3 - phi - M**2 * lap_phi

    mu_hat = fft(mu)
    nonlinear1 = -k2 * mu_hat
    nonlinear1_real = ifft(nonlinear1).real

    # First ETD2 stage
    a = phi + dt/2 * nonlinear1_real

    # Second half-step: compute new nonlinear term
    a_hat = fft(a)
    lap_a_hat = -k2 * a_hat
    lap_a = ifft(lap_a_hat).real
    mu_a = a**3 - a - M**2 * lap_a

    mu_a_hat = fft(mu_a)
    nonlinear2 = -k2 * mu_a_hat

    # ETD2 update
    phi_hat_new = E * phi_hat + phi_func * nonlinear1 + E2 * phi_func2 * nonlinear2
    phi = ifft(phi_hat_new).real

    # Optional: save or plot every few steps
    if step % save_every == 0:
        plt.clf()
        plt.plot(x, phi)
        plt.title(f"Step {step}")
        plt.pause(0.01)

plt.show()