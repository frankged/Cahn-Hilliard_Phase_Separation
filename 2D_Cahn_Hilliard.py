import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fft import fft2, ifft2, fftfreq

N = 4 # Number of grid points
BOUNDARY = 'periodic' # Boundary condition type
h = 1.0 / N # Step size
M = 1.0 # Mobility constant
kappa = 1.0 # Gradient energy coefficient

# Initialize randomly probability field phi as a 2D array
phi_0 = np.random.uniform(-1,1, size = (N,N))

#for FFT:
# Wavenumbers
k = 2 * np.pi * fftfreq(N, d=h)
KX, KY = np.meshgrid(k, k, indexing='ij')
k2 = KX**2 + KY**2


# Define the time span and initial condition
t_span = (0, 10)  # Start and end time
phi_0 = phi_0.flatten()  # Flatten to 1D array for solve_ivp
t_eval = np.linspace(t_span[0], t_span[1], 10)  # Time points to evaluate
pbar = tqdm(total=len(t_eval))      
def laplacian_2D(phi, h, N, boundary='periodic'):
    # Reshape to 2D
    phi = phi.reshape((N, N))
    
    # Periodic Laplacian
    lap = (
        (np.roll(phi, -1, axis=0) + np.roll(phi, 1, axis=0) - 2 * phi) / h**2
      + (np.roll(phi, -1, axis=1) + np.roll(phi, 1, axis=1) - 2 * phi) / h**2
    )
    
    # Return flattened
    return lap.ravel()

def FFT_laplacian_2D(phi, h, N, boundary='periodic'):
    phi = phi.reshape((N, N))
    phi_hat = fft2(phi)
    #laplacian operator in Fourier space
    lap_phi_hat = -k2 * phi_hat

    #inverse fft
    lap_phi = ifft2(lap_phi_hat).real
    return lap_phi.flatten()

def f(t, phi):
    pbar.n = np.searchsorted(t_eval, t)
    pbar.refresh()
    del_phi = laplacian_2D(phi, h, N, boundary=BOUNDARY) 
    return M * (-del_phi + laplacian_2D(phi**3, h, N, boundary=BOUNDARY) - kappa * laplacian_2D(del_phi, h, N, boundary=BOUNDARY))
# Solve the Cahn-Hilliard equation using solve_ivp
# solution = solve_ivp(f, t_span, phi_0, t_eval=t_eval)
# print("Solution shape:", solution.y.shape)

def f_fft(t, phi):
    pbar.n = np.searchsorted(t_eval, t)
    pbar.refresh()
    del_phi = FFT_laplacian_2D(phi, h, N, boundary=BOUNDARY) 
    return M * (-del_phi + FFT_laplacian_2D(phi**3, h, N, boundary=BOUNDARY) - kappa * FFT_laplacian_2D(del_phi, h, N, boundary=BOUNDARY))

solution = solve_ivp(f_fft, t_span, phi_0, t_eval=t_eval)
print("Solution shape:", solution.y.shape)


