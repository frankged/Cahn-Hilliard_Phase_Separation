import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

N = 100 # Number of grid points
BOUNDARY = 'periodic' # Boundary condition type
h = 1.0 / N # Step size
M = 1.0 # Mobility constant
kappa = 1.0 # Gradient energy coefficient

# Initialize randomly probability field phi as a 1D array
phi_0 = np.random.uniform(-1,1, size = N)

# Initialize 1D Laplacian operator
main_diag = -2 * np.ones(N)
off_diag = np.ones(N - 1)

L_1 = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1) 
if BOUNDARY == 'periodic':
    # Add periodic connections
    L_1[0, -1] = 1
    L_1[-1, 0] = 1

def f(t, phi):
    return M*(-L_1 @ phi + L_1 @ (phi**3) - kappa * L_1 @ (L_1 @ phi))

# Define the time span and initial condition
t_span = (0, 100)  # Start and end time
phi_0 = phi_0.flatten()  # Flatten to 1D array for solve_ivp
t_eval = np.linspace(t_span[0], t_span[1], 100)  # Time points to evaluate      
# Solve the Cahn-Hilliard equation using solve_ivp
solution = solve_ivp(f, t_span, phi_0, t_eval=t_eval)
print("Solution shape:", solution.y.shape)
# Plot the results
plt.figure(figsize=(10, 6))
for i in range(0, solution.y.shape[1], max(1, solution.y.shape[1] // 10)):  # Plot ~10 time snapshots
    plt.plot(np.arange(N), solution.y[:, i], label=f't={solution.t[i]:.2f}')
plt.xlabel('Spatial index')
plt.ylabel('Probability Field (phi)')
plt.title('Cahn-Hilliard Equation Solution Over Time')
plt.legend()
plt.grid()
plt.show()