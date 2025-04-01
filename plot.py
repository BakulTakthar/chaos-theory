import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Lorenz system function
def lorenz(t, state, sigma=10, beta=8/3, rho=28):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Initial conditions
state0 = [1.0, 1.0, 1.0]

# Time span
time_span = (0, 50)
time_eval = np.linspace(time_span[0], time_span[1], 10000)

# Solve the system
solution = solve_ivp(lorenz, time_span, state0, t_eval=time_eval)
t, x, y, z = solution.t, solution.y[0], solution.y[1], solution.y[2]

# Create figure and subplots
fig = plt.figure(figsize=(12, 6))
grid = fig.add_gridspec(3, 2, width_ratios=[1, 1.5], height_ratios=[1, 1, 1])  # Define grid layout


# X vs Time
ax1 = fig.add_subplot(grid[0,0])
ax1.plot(t, x, color='r')
ax1.set_title("X vs Time")
ax1.set_xlabel("Time")
ax1.set_ylabel("X")

# Y vs Time
ax2 = fig.add_subplot(grid[1,0])
ax2.plot(t, y, color='g')
ax2.set_title("Y vs Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Y")

# Z vs Time
ax3 = fig.add_subplot(grid[2,0])
ax3.plot(t, z, color='b')
ax3.set_title("Z vs Time")
ax3.set_xlabel("Time")
ax3.set_ylabel("Z")

# 3D Lorenz Attractor
ax4 = fig.add_subplot(grid[:,1], projection='3d')
ax4.plot(x, y, z, lw=0.5, color='blue')
ax4.scatter(x[0], y[0], z[0], color='red', s=50, label='Start')
ax4.set_xlabel("X Axis")
ax4.set_ylabel("Y Axis")
ax4.set_zlabel("Z Axis")
ax4.set_title("Lorenz Attractor")

# Adjust layout
plt.tight_layout()
plt.show()


## to see what is the divergence of the DE system

div = lorenz(0, state0)

print(div)