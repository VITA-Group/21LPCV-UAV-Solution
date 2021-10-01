# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]

        # Q: the noise matrix Q of the system
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x(t)' = A*x(t-1) + B*u(k): assuming a constant velocity model, thus B=0
        mean = np.dot(self._motion_mat, mean)
        # P(t)' = A*P(t-1)*A^T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """

        # Covariance of the noise in measurement
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]

        # R: the noise matrix of the detector
        innovation_cov = np.diag(np.square(std))

        # H*x(t)': project the mean vector of the object's state distribution to measurement space
        mean = np.dot(self._update_mat, mean)

        # H*P(t)'*H^T: project the covariance matrix of the object's state distribution to measurement space
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """

        # H*x(t)', H*P(t)'*H^T + R
        projected_mean, projected_cov = self.project(mean, covariance)

        '''
        The Cholesky decomposition of a matrix:  A = L * L^T or A = U * U^T of a positive-definite matrix A.
        The return value can be directly used as the first parameter to cho_solve.
        '''
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)

        '''
        Solve the linear equations A*x = b, given the Cholesky factorization of A.
        '''
        # K(t) = P(t)'*H^T ( H*P(t)'*H^T + R )^{-1}
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T

        # y = z - H*x(t)'
        innovation = measurement - projected_mean

        # x(t) = x(t)' + K(t)*y
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        '''
        P(t) = P(t)' - K(t) * (HP(t)'H^T + R) * K(t)^T
             = P(t)' - P(t)'*H^T ( H*P(t)'*H^T + R )^{-1} * ( H*P(t)'*H^T + R ) * K(t)^T
             = P(t)' - P(t)'*H^T * K(t)^T
             = P(t)' - P(t)'*(K(t)*H)^T
             = P(t)' - K(t)*H*P(t)'
        '''

        # P(t) = (I - K(t)*H)*P(t)'
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        '''
        Return the Cholesky decomposition: A = L * L^T, where L is lower-triangular A is positive-definite.
        In addition, only the lower-triangular and diagonal elements of a are used.
        Only L is actually returned.
        The Cholesky decomposition is often used as a fast way of solving: A * x = b,
        by first solving L * y =b, and then for x in L^T x = y.
        '''
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        '''
        Solve the equation A * x = b for x, assuming A is a triangular matrix.
        '''
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)

        '''
        (Squared) Mahalanobis distance: d^(1)(i,j) = (d_j - y_i)^T * S_i^{-1} * (d_j - y_i).
        The projection of the i-th track distribution into measurement space is denoted by (y_i,S_i),
        and the j-th bounding box detection is denoted by d_j.
        The Mahalanobis distance takes state estimation uncertainty into account by measuring how many standard deviations the detection is away from the mean track location.
        Further, if using this metric, it is possible to exclude unlikely associations,
        by thresholding the Mahalanobis distance at a 95% confidence interval computed from the inverse Chi-square distribution.
        The decision is denoted with an indicator: b_{i,j}^(1) = 1[d^(1)(i,j) <= t^(1)].
        It evaluates to 1 if the association between the i-th track and j-th detection is admissible.
        For this 4 dimensional measurement space, the corresponding Mahalanobis threshold is t^(1) = 9.4877.
        '''
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
