import numpy as np


def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    sqdist = (
        np.sum(x1**2, axis=1).reshape(-1, 1)
        + np.sum(x2**2, axis=1)
        - 2 * np.dot(x1, x2.T)
    )
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)


class GaussianProcessRegressor:
    def __init__(self, kernel=rbf_kernel, length_scale=1.0, sigma_f=1.0, sigma_n=1e-8):
        self.kernel = kernel
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def fit(self, X_train: np.ndarray[np.float64], Y_train: np.ndarray[np.float64]):
        self.X = X_train
        self.Y = Y_train

        # Compute the kernel matrix K(X, X) + sigma_n^2 * I
        self.K = self.kernel(
            X_train, X_train, self.length_scale, self.sigma_f
        ) + self.sigma_n**2 * np.eye(len(X_train))

        # Perform Cholesky decomposition of K
        self.L = np.linalg.cholesky(self.K)

        # Solve for alpha using Cholesky factors
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, Y_train))

    def predict(self, x_test: np.ndarray[np.float64]) -> tuple[np.float64, np.float64]:
        K_star = self.kernel(self.X, x_test, self.length_scale, self.sigma_f)
        K_2star = self.kernel(
            x_test, x_test, self.length_scale, self.sigma_f
        ) + self.sigma_n**2 * np.eye(len(x_test))

        # Predictive mean
        mu_star = K_star.T.dot(self.alpha)

        # Solve for v
        v = np.linalg.solve(self.L, K_star)

        # Predictive covariance
        cov_star = K_2star - v.T.dot(v)

        return mu_star, np.diag(cov_star)


# Main script
if __name__ == "__main__":
    # Training data: 7 samples, each a 3x3 matrix flattened to a 9-dimensional vector
    X_train = np.array(
        [
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
            [1, 2, 3, 1, 1, 1, 1, 1, 1],
        ]
    )  # Reshape to (n_samples, K*3)
    y_train = np.array([1, 3, 2, 5, 4, 6, 5])

    # Test data: 1 sample, a 3x3 matrix flattened to a 9-dimensional vector
    X_test = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]])  # Reshape to (n_samples, K*3)

    # Create and train the GPR model
    gpr = GaussianProcessRegressor(length_scale=1.0, sigma_f=1.0, sigma_n=0.1)
    gpr.fit(X_train, y_train)

    # Make predictions
    y_pred, sigma = gpr.predict(X_train[:1])

    # Print predictions and uncertainties
    print("Predictions:", y_pred)
    print("Uncertainties:", sigma)
