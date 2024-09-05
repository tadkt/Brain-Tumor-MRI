import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def f(x,y):
    "Objective Function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

# Compute and plot the function in 3D within [0,5]x[0,5]
x, y = np.array(np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100)))

z = f(x, y)
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

plt.figure(figsize=(8,6))
plt.imshow(z, extent=[0,5,0,5], origin='lower', cmap='viridis', alpha=0.5)
plt.colorbar()
plt.plot([x_min], [y_min], marker='x', markersize=5, color='white')
contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
plt.clabel(contours, inline=True, fontsize=8, fmt='%.0f')

n_particles = 20
X = np.random.rand(2, n_particles) * 5
V = np.random.randn(2, n_particles) * 0.10
plt.scatter(X[0], X[1], c='red', s=50, alpha=0.8, label='Particles')

# Plot velocity vectors
plt.quiver(X[0], X[1], V[0], V[1], color='white', scale=5, width=0.005)

plt.legend()
plt.title('Particle Swarm Optimization - Initial State')
plt.show()

pbest = X
pbest_obj = f(X[0], X[1])

gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()

c1 = c2 = 0.1
w = 0.8

# One iteration
r = np.random.rand(2)
V = w * V + c1*r[0]*(pbest - X) + c2*r[1]*(gbest.reshape(-1,1)-X)
X = X + V
obj = f(X[0], X[1])
pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
pbest_obj = np.array([pbest_obj, obj]).max(axis=0)
gbest = pbest[:, pbest_obj.argmin()]
gbest_obj = pbest_obj.min()