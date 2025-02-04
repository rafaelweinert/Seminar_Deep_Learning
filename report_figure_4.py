#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import mean
from numpy import std
from scipy.optimize import root


#%% define loss functions and gradients

def loss_function(x):
    return (x**7-1)**2 + np.exp(-(x-1)**2)


def gradient(x):
    return 14*(x**7-1)*x**6 + np.exp(-(x-1)**2) * (-2*(x-1))

# %% SGD
x = np.linspace(-1.2, 1.2, 1000)
y = loss_function(x)
y_grad = gradient(x)


# Parameters
n_trajectories = 100
n_steps = 5000
learning_rate = 0.01
noise_scale = 1

trajectories = np.empty((n_trajectories, n_steps + 1, 1))

# SGD for n_trajectories
for j in range(n_trajectories):
    # Initialize point and trajectory
    theta = np.random.uniform(min(x), max(x), size=1)
    trajectory = np.empty((n_steps + 1, 1))
    trajectory[0] = theta

    for i in range(1, n_steps + 1):
        grad = gradient(theta)
        #noise = np.random.exponential(scale=noise_scale, size=theta.shape) * np.random.choice([-1, 1], size=theta.shape)
        #noise = np.random.standard_cauchy(size=theta.shape) * noise_scale
        #noise = np.random.beta(2,2, size=theta.shape) * noise_scale * np.random.choice([-1, 1], size=theta.shape)
        residual = loss_function(theta) - np.mean([loss_function(theta + np.random.uniform(-0.1, 0.1)) for _ in range(10)])
        noise = residual * np.random.normal(0, scale=noise_scale, size=theta.shape)
        #noise = np.random.normal(0, scale=noise_scale, size=theta.shape)
        theta = theta - learning_rate * (grad + noise)
        trajectory[i] = theta

    trajectories[j] = trajectory
    

trajectories = np.array(trajectories)
trajectories_reshaped = trajectories.reshape(-1,1)


#%% look at critical points

# find critical points where gradient is zero
def gradient_zero_points(grad_func, initial_guesses):
    """
    Finds the points where the gradient is zero.

    Args:
        grad_func (function): A function representing the gradient.
                              It should take an array-like input and return an array-like gradient.
        initial_guesses (list): A list of initial guesses for the root-finding algorithm.

    Returns:
        list: Points where the gradient is zero.
    """
    zero_points = []
    for guess in initial_guesses:
        res = root(grad_func, guess)
        if res.success:
            zero_points.append(res.x.item())
    return zero_points
critical_points = gradient_zero_points(gradient, np.linspace(min(x), max(x), 10))

# drop duplicates of critical points and points that are very close to each other
def drop_close_points(points, tolerance=1e-1):
    # only points that are smaller than 1e5
    p = [point for point in points if abs(point) < 1e5]
    p = sorted(set(p))
    filtered_points = [p[0]]
    for point in p[1:]:
        if (abs(point - filtered_points[-1]) > tolerance):
            filtered_points.append(point)
    return filtered_points

critical_points = drop_close_points(critical_points)



fig, ax = plt.subplots(figsize=(8, 6))

ax.set_yscale('log')
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('Loss function', color='b', fontsize=14)
ax.plot(x, y, 'b-', label='Loss function')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('One-dimensional function and SGD trajectories', fontsize=16, pad=20)


ax_r = ax.twinx()
ax_r.set_xlabel('x', fontsize=14)
ax_r.set_ylabel('Density of SGD trajectories', color='g', fontsize=14)

#plt.savefig('Presentation_plots/Long_run_One_dimensional_function.png', dpi=300, bbox_inches='tight')
for point in critical_points:
    ax.axvline(point, color='r', linestyle='--')
#plt.savefig('Presentation_plots/Long_run_One_dimensional_function_critical_points.png', dpi=300, bbox_inches='tight')

ax_r.hist(trajectories_reshaped[:, 0], bins=100, density=True, alpha=0.7, color='g')

#plt.savefig('Presentation_plots/Long_run_One_dimensional_function_histogram.png', dpi=300, bbox_inches='tight')
# only show the range of x for -1 tp 1
plt.show()

