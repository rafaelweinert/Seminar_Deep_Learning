#%% import libraries
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# Parameters
n = 100          # Number of samples
d = 200     # Dimension of the samples
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



plt.figure(figsize=(10, 6))
pp = 0
plt.plot(np.arange(iterations)[pp:], errors[pp:], color='blue')
plt.xlabel('Iterations (Time)', fontsize=14)
plt.ylabel(r'Error $\|\theta_t - \theta_*\|^2$' , fontsize = 14)
plt.yscale('log')
plt.xscale('log')
plt.tick_params(axis='both', which='major', labelsize=14)
plt.title('Error of SGD (log-log scale)', fontsize=16)

# get the smalles eigenvalue that is bigger than 0.001
min_eigenvalue = np.min(eigenvalues[eigenvalues > 0.000000001])
plt.axvline(x=1/min_eigenvalue, color='orange', linestyle='--', linewidth=2, label='Transition')



plt.show()
