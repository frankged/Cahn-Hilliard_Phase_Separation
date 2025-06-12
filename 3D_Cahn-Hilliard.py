import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_

import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftfreq
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import vtk
import pyvista as pv
import os

# Grid parameters
Nx, Ny, Nz = 64, 64, 64
Lx, Ly, Lz = 50.0, 50.0, 50.0
dx = Lx / Nx
dy = Ly / Ny
dz = Lz / Nz

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
z = np.linspace(0, Lz, Nz, endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Time parameters
dt = 0.01
n_steps = 1000
save_every = 10  # Save every 10 steps

# Physical parameter
epsilon = 1.0

# Wavenumbers
kx = 2 * np.pi * fftfreq(Nx, d=dx)
ky = 2 * np.pi * fftfreq(Ny, d=dy)
kz = 2 * np.pi * fftfreq(Nz, d=dz)

KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
k2 = KX**2 + KY**2 + KZ**2
k4 = k2**2

# Initial condition
np.random.seed(0)
phi = 0.01 * (2 * np.random.rand(Nx, Ny, Nz) - 1)

# Precompute ETD2 coefficients
L_hat = -k4
E = np.exp(L_hat * dt)
E2 = np.exp(L_hat * dt / 2)

L_hat[L_hat == 0] = 1e-20
inv_L_hat = 1 / L_hat
phi_func = (E - 1) * inv_L_hat
phi_func2 = (E2 - 1) * inv_L_hat

# For time lapse: store snapshots
phi_snapshots = []

# Time loop
for step in range(n_steps):

    # First half-step
    phi_hat = fftn(phi)
    lap_phi_hat = -k2 * phi_hat
    lap_phi = ifftn(lap_phi_hat).real
    mu = phi**3 - phi - epsilon**2 * lap_phi

    mu_hat = fftn(mu)
    nonlinear1 = -k2 * mu_hat
    nonlinear1_real = ifftn(nonlinear1).real

    # ETD2 stage
    a = phi + dt/2 * nonlinear1_real

    a_hat = fftn(a)
    lap_a_hat = -k2 * a_hat
    lap_a = ifftn(lap_a_hat).real
    mu_a = a**3 - a - epsilon**2 * lap_a

    mu_a_hat = fftn(mu_a)
    nonlinear2 = -k2 * mu_a_hat

    # ETD2 update
    phi_hat_new = E * phi_hat + phi_func * nonlinear1 + E2 * phi_func2 * nonlinear2
    phi = ifftn(phi_hat_new).real

    # Save snapshots
    if step % save_every == 0:
        print(f"Saving snapshot at step {step}")
        phi_snapshots.append(phi.copy())

# ---- Animate 3D isosurface ---- #


# Set up figure
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Choose isovalue
iso_value = 0.0

# Initial empty plot (will update in animation)
verts, faces, normals, values = measure.marching_cubes(phi_snapshots[0], level=iso_value, spacing=(dx, dy, dz))
mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap='Spectral', lw=0.5)

# Axes limits
ax.set_xlim(0, Lx)
ax.set_ylim(0, Ly)
ax.set_zlim(0, Lz)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Update function for animation
def update(frame):
    global mesh
    if mesh is not None:
        mesh.remove()
    verts, faces, normals, values = measure.marching_cubes(phi_snapshots[frame], level=iso_value, spacing=(dx, dy, dz))
    mesh = ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2], triangles=faces, cmap='Spectral', lw=0.5)
    ax.set_title(f"Step {frame * save_every}")
    return mesh,

# Animate
ani = animation.FuncAnimation(fig, update, frames=len(phi_snapshots), interval=200, blit=False)

plt.show()
ani.save("cahn_hilliard_3d.mp4", fps=5)

