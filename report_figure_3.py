#%% import libraries
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
n = 100          # Number of samples
d = 2       # Dimension of the samples
iterations = 100000   # Number of SGD iterations
learning_rate = 0.1
noise = 0.2


# Variance reduction techniques: decaying step size and averaging
decaying_step_size = False
averaging = False

# Create a covariance matrix with eigenvalues decaying as a power law
eigenvalues = np.array(object=[1 / (i + 1)**1.25 for i in range(d)])  # decay as power law 1 / (i + 1)^2
# eigenvalues[100:] = 0  # Set the last 100 eigenvalues to zero
Q = np.random.randn(d, d)
Q, _ = np.linalg.qr(Q)  # create an orthogonal matrix Q
cov_matrix = Q @ np.diag(eigenvalues) @ Q.T  # covariance matrix with decaying eigenvalues

# Generate random data from Gaussian distribution with this covariance matrix
X = np.random.multivariate_normal(np.zeros(d), cov_matrix, size=n)
theta_star = np.zeros(d) #np.random.randn(d)  # True parameters (randomly chosen)#
y = X @ theta_star + noise * np.random.randn(n)  # Add some noise to the outputs

# Initialize parameters for the linear model
theta_0 = np.random.normal(size=d, scale = 1.5, loc = 1)

def sgd(decaying_step_size=False, averaging=False):
    np.random.seed(0)
    theta = theta_0.copy()

    # Store thetas at each iteration
    thetas = []
    # Record the error at each iteration
    errors = []

    for i in range(iterations):
        thetas.append(theta.copy())
        
        # Randomly pick a sample for SGD
        idx = np.random.randint(0, n)
        x_i = X[idx]
        y_i = y[idx]
        
        # Calculate the prediction and error
        prediction = np.dot(x_i, theta)
        error = prediction - y_i
        
        # Square loss gradient
        grad = 2 * error * x_i
        
        # Update theta using gradient descent
        if decaying_step_size:
            step_size = 1/(100+(learning_rate*i)**1.01)#1 / np.sqrt(i + 1)#(3+i**1.005)#
        else: 
            step_size = learning_rate
        theta -= step_size * grad
        
        # Calculate and store the error (mean squared error)
        mse = np.mean((X @ theta - y) ** 2)
        
        print(f"Iteration {i + 1}, MSE: {mse}")
        
        errors.append(mse)
    thetas = np.array(thetas)

    if averaging:
        thetas = np.cumsum(thetas, axis=0) / np.arange(1, len(thetas) + 1)[:, None]

        #thetas = np.cumsum(thetas, axis=0) / np.array([np.arange(1, len(thetas) + 1), np.arange(1, len(thetas) + 1)]).T

    return thetas, errors



thetas, errors = sgd()
thetas_stepsize, errors_stepsize = sgd(decaying_step_size=True)
thetas_averaging, errors_averaging = sgd(averaging=True)



if d <= n:
    theta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

else:
    X_pinv = X.T @ np.linalg.inv(X @ X.T)  # Pseudo-inverse of X
    theta_hat = X_pinv @ y + (np.eye(d) - X_pinv @ X) @ theta_0




fig, ax = plt.subplots(figsize=(6, 6))

# plot two random dimensions for theta for the three cases
a, b = 0,1#np.random.choice(d, 2, replace=False)

theta_1 = np.linspace(-2.1, 2.1, 100)
theta_2 = np.linspace(-2.1, 2.1, 100)
T_1, T_2 = np.meshgrid(theta_1, theta_2)
L = 0.5 * (T_1**2 + T_2**2)


thetas = np.array(thetas)
thetas_stepsize = np.array(thetas_stepsize)
thetas_averaging = np.array(thetas_averaging)

contours = ax.contour(T_1, T_2, L, levels=20, cmap='viridis')
ax.plot(thetas[:, a], thetas[:, b], color='blue', alpha=0.7, label='No variance reduction')
ax.plot(thetas_stepsize[:, a], thetas_stepsize[:, b], color='red', alpha=0.7, label='Stepsize decay')
ax.plot(thetas_averaging[:, a], thetas_averaging[:, b], color='green', alpha=0.7, label='Time averaging')

ax.scatter(theta_hat[a], theta_hat[b], color='black', s=100, label=r'$\theta_*$')

ax.set_xlabel(r'Dimension {}'.format(a), fontsize=14)
ax.set_ylabel(r'Dimension {}'.format(b), fontsize=14)
ax.legend(fontsize=14, frameon=True, facecolor='white', framealpha=1)
ax.set_title('SGD Trajectories for different variance reduction techniques', fontsize=16, pad=20)

#plt.savefig('SDE_for_SGD/SGD_Trajectories.png', dpi=300, bbox_inches='tight')
plt.show()


