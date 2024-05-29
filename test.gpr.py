# pylint: disable=invalid-name

from typing import Generic, TypeVar
import numpy as np

HyperParameter = float

PredictionResult = TypeVar("PredictionResult")


def rbf_kernel(
    x: np.ndarray[np.float64],
    x_star: np.ndarray[np.float64],
    length_scale: HyperParameter = 1.0,
    sigma_f: HyperParameter = 1.0,
) -> float:
    """
    Dimension reduction function, mapping the input into higher dimension space

    RBF kernel

    Returns:

    1. Calculate Euclidean distance between `x`, `x*`, `||x - x*||^2`
    2. Calculate the RBF kernel, `exp(-||x - x*||^2 / (2 * σ^2))`
    3. Returns the RBF kernel matrix, which represents the similarity between `x` and `x*`
    """
    sqrt_dist = (
        np.sum(x**2, axis=1).reshape(-1, 1)
        + np.sum(x_star**2, axis=1)
        - 2 * np.dot(x, x_star.T)
    )
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqrt_dist)


class GaussianProcessRegressor(Generic[PredictionResult]):
    """
    Gaussian Process Regression (GPR) function

    Assumption:

    - `y_i` = `f(x_i) + ℇ_i`, (where `ℇ_i` ~ `N(0, σ_n^2)`)
        - `f(x_i)` = target learning function
        - `ℇ_i` = gaussian noise
    - `f(x)` ∼ `GP(m(x),k(x,x*))`
    - `x`: input
    - `x*`: target input
    - `f(x)`: target learning function ~ GP
    - `m(x)`: mean function = `E[f(x)]`
    - `k(x,x*)`: covariance function(kernel)
        = `E[(f(x) - m(x))(f(x*) - m(x*))]`,

        defines the similarity between `x` and `x*`

    Training data:

    1. `X` = `{x_1, x_2, ..., x_N}` - training data points(input)
    2. `Y` = `[y_1, y_2, ..., y_N]T` - training result(output)

    Prior distribution:

    `f(X)` = `[f(x_1), f(x_2), ..., f(x_N)]T`

    follows the multivariate Gaussian distribution -> `f(X)` ~ `N(m(X), K(X, X))`
        - mean vector: `m(X)` = `[m(x_1), m(x_2), ..., m(x_N)]T`
        - covariance matrix(NxN): `K(X, X)` = `[k(x_1, x_1), k(x_1, x_2), ..., k(x_1, x_N)]`

    Noise Model:
    - `Y` = `f(X) + ℇ`, where (`ℇ` ~ `N(0, σ_n^2)`)
    - `y_i` = `f(x_i) + ℇ_i` where (`ℇ_i` ~ `N(f(x_i), σ_n^2)`)
    """

    def __init__(
        self,
        kernel=rbf_kernel,
        length_scale: HyperParameter = 1.0,
        sigma_f: HyperParameter = 1.0,
        sigma_n: HyperParameter = 1e-8,
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n

    def fit(self, X_train: np.ndarray[np.float64], Y_train: np.ndarray[np.float64]):
        """
        Fit the GPR model into `{X_train}, {Y_train}`
        """
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

    def predict(
        self, x_test: np.ndarray[np.float64]
    ) -> tuple[PredictionResult, np.float64]:
        """
        Predict the mean and variance for x_test

        Returns:
            mu_star: mean (prediction)
            cov_star: covariance (uncertainty)
        """
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
    y_pred, sigma = gpr.predict(X_test)
    y_pred2, sigma2 = gpr.predict(X_train[:1])

    # Print predictions and uncertainties
    print("Predictions:", y_pred)
    print("Uncertainties:", sigma)

    print("Predictions2:", y_pred2)
    print("Uncertainties2:", sigma2)
