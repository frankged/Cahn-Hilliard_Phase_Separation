import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq


#calculate characteristic length scale
def characteristic_length_scale(phi):
    """Calculate the characteristic length scale of the phase field."""
    # Compute the Fourier transform of phi
    phi_hat = fft2(phi)
    
    # Compute the structure factor
    structure_factor = np.abs(phi_hat)**2
    
    # Calculate the wavenumber
    kx = 2 * np.pi * fftfreq(N, d=dx)
    ky = 2 * np.pi * fftfreq(N, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Calculate the characteristic length scale
    length_scale = np.sum(structure_factor) / np.sum(structure_factor * (KX**2 + KY**2))
    
    return length_scale

# Grid parameters
N = 128      # Grid size N x N (try 128, 256, 512)
L = 50.0     # Domain size
dx = L / N
dy = L / N
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Time parameters
dt = 0.01
n_steps = 10000
save_every = 100

# Physical parameter
epsilon = 1.0

# Wavenumbers
kx = 2 * np.pi * fftfreq(N, d=dx)
ky = 2 * np.pi * fftfreq(N, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')
k2 = KX**2 + KY**2
k4 = k2**2

# Initial condition: random noise
np.random.seed(0)
phi = 0.01 * (2 * np.random.rand(N, N) - 1)

# Precompute ETD2 coefficients
L_hat = -k4
E = np.exp(L_hat * dt)
E2 = np.exp(L_hat * dt / 2)

# ETD denominators (avoid divide by 0)
L_hat[L_hat == 0] = 1e-20
inv_L_hat = 1 / L_hat
phi_func = (E - 1) * inv_L_hat
phi_func2 = (E2 - 1) * inv_L_hat

# Time loop
for step in range(n_steps):

    # First half-step: compute nonlinear term
    phi_hat = fft2(phi)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = ifft2(lap_phi_hat).real
    mu = phi**3 - phi - epsilon**2 * lap_phi

    mu_hat = fft2(mu)
    nonlinear1 = -k2 * mu_hat
    nonlinear1_real = ifft2(nonlinear1).real

    # First ETD2 stage
    a = phi + dt/2 * nonlinear1_real

    # Second half-step: compute new nonlinear term
    a_hat = fft2(a)
    lap_a_hat = -k2 * a_hat
    lap_a = ifft2(lap_a_hat).real
    mu_a = a**3 - a - epsilon**2 * lap_a

    mu_a_hat = fft2(mu_a)
    nonlinear2 = -k2 * mu_a_hat

    # ETD2 update
    phi_hat_new = E * phi_hat + phi_func * nonlinear1 + E2 * phi_func2 * nonlinear2
    phi = ifft2(phi_hat_new).real

    # Optional: save or plot every few steps
    if step % save_every == 0:
        plt.clf()
        plt.imshow(phi, extent=(0, L, 0, L), cmap='RdBu', origin='lower')
        plt.colorbar()
        plt.title(f"Step {step}, Characteristic Length Scale: {characteristic_length_scale(phi):.2f}")
        plt.pause(0.01)

plt.show()