#%%
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

#%matplotlib inline

# Define squared loss function
def squared_loss(w, X, y):
    return np.sum((X @ w - y) ** 2).astype(np.float64)

def gradient(w, X, y):
    return 2 * X.T @ (X @ w - y).astype(np.float64)

# Generate sample data: X (2x2 matrix), y (2x1 vector)
X = np.array([[5, -2]])
y = np.array([1])

# Generate a grid for w1 and w2
w1 = np.linspace(-1, 3, 100, dtype=np.float64)
w2 = np.linspace(-1, 3, 100, dtype=np.float64)
W1, W2 = np.meshgrid(w1, w2)

# Compute the loss for each combination of w1 and w2
loss = np.zeros_like(W1)
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w = np.array([W1[i, j], W2[i, j]])
        loss[i, j] = squared_loss(w, X, y)



theta_0s = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [2.5, 2.5]])


#%% Gradient descent for various starting points theta_0

# Plot the loss surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
lr = np.float32(0.001)
n_iter = 100


# Perform gradient descent for each starting point
for theta_0 in theta_0s:
    theta = theta_0.copy()
    trajectory = [theta.copy()]
    for _ in range(n_iter):
        # Stochastic gradient descent update
        theta -= lr * gradient(theta, X, y) + np.sqrt(lr) * np.random.normal(0, 0.3, size=theta.shape)
        trajectory.append(theta.copy())
    trajectory = np.array(trajectory)

    # Plot the trajectory on the loss surface
    ax.plot(trajectory[:, 0], trajectory[:, 1], [squared_loss(t, X, y) for t in trajectory], 
            marker='.', label=f'$\\theta_0 = [{theta_0[0]}, {theta_0[1]}], \\theta^* = [{np.round(trajectory[-1][0], 2)}, {np.round(trajectory[-1][1], 2)}]$')
    ax.legend()
# ax.plot(trajectory[:, 0], trajectory[:, 1], [squared_loss(t, X, y) for t in trajectory], marker='o', label=f'$\\theta_0 = {theta_0}$')
ax.plot_surface(W1, W2, loss, cmap='viridis', alpha=0.5)
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('Loss')
ax.set_title('Trajectories of SGD')


zero_loss = np.where(loss <= 0.00000001)
ax.plot(W1[zero_loss], W2[zero_loss], 0)


plt.show(block=False)



#%% calculate the orthogonal projection of theta_0 into I
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, loss, cmap='viridis', alpha=0.5)

X_dagger = np.linalg.pinv(X)

for theta_0 in theta_0s:
    theta = theta_0.copy()
    
    # Compute the orthogonal projection of theta_0 into I
    theta_star = X_dagger @ y + (np.eye(X.shape[1]) - X_dagger @ X) @ theta
    print(f"theta_0 = {theta_0}, theta* = {theta_star}")

    # Plot the orthogonal projection and the original point, ensuring shared color and legend
    ax.plot(
        [theta_0[0], theta_star[0]], 
        [theta_0[1], theta_star[1]], 
        [squared_loss(theta_0, X, y), squared_loss(theta_star, X, y)],
        marker='o', label=f'$\\theta_0 = [{theta_0[0]}, {theta_0[1]}], \\theta^* = [{np.round(theta_star[0], 2)}, {np.round(theta_star[1], 2)}]$'
    )

ax.legend()
ax.set_xlabel('$w_1$')
ax.set_ylabel('$w_2$')
ax.set_zlabel('Loss')
ax.set_title('Orthogonal Projection of $\\theta_0$ into $I$')

zero_loss = np.where(loss <= 0.00000001)
ax.plot(W1[zero_loss], W2[zero_loss], 0)

plt.show()


# %%
