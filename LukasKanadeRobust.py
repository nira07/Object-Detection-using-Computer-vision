import numpy as np
from scipy.interpolate import RectBivariateSpline

# Get weight matrix for M-Estimator
# Default a given by https://www.real-statistics.com/descriptive-statistics/m-estimators/
def getWeightMatrix(e, a=None, mtype="huber", diag_matrix=False):
    if mtype == "huber":
        if a is None:
            a = 1.339
        weights = np.tile(a, e.shape[0]) / np.abs(e)
    elif mtype == "tukey":
        if a is None:
            a = 4.685
        weights = (np.ones_like(e) - (e/a)**2)**3
    elif mtype == "none":
        weights = np.ones_like(e)

    np.clip(weights, 0, 1, out=weights)

    if diag_matrix:
        return np.diag(weights)

    return weights


def LucasKanadeAffineRobust(It, It1, rect, mtype="huber"):
    """
    Lucas-Kanade Affine Robust using M-estimator.
    Inputs:
        It: Template image
        It1: Current image
        rect: Current bounding box [x1, y1, x2, y2]
        mtype: Type of robust loss function ("huber", "tukey", "none")
    Output:
        p: Affine warp parameters (6x1 vector)
    """

    # Threshold and max iterations
    threshold = 0.01875
    maxIters = 100

    x1, y1, x2, y2 = rect
    p = np.zeros((6, 1), dtype=np.float64)

    # Create splines for interpolation
    height, width = It.shape
    y = np.arange(height)
    x = np.arange(width)
    splineIt = RectBivariateSpline(y, x, It)
    splineIt1 = RectBivariateSpline(y, x, It1)

    # Generate grid of coordinates for template
    coordsX, coordsY = np.meshgrid(
        np.linspace(x1, x2, int(x2 - x1)),
        np.linspace(y1, y2, int(y2 - y1)),
    )

    # Compute gradients of the template
    Itx = splineIt.ev(coordsY, coordsX, dx=1, dy=0)
    Ity = splineIt.ev(coordsY, coordsX, dx=0, dy=1)

    # Flatten coordinates
    flat_coordsX = coordsX.flatten()
    flat_coordsY = coordsY.flatten()
    num_points = flat_coordsX.size

    # Compute Jacobian matrix
    J = np.zeros((num_points, 6), dtype=np.float64)
    J[:, 0] = flat_coordsX
    J[:, 1] = flat_coordsY
    J[:, 2] = 1
    J[:, 3] = 0
    J[:, 4] = 0
    J[:, 5] = 0

    for _ in range(maxIters):
        # Warp the current image
        warpedI = splineIt1.ev(flat_coordsY + p[3] * flat_coordsX + p[4] * flat_coordsY + p[5],
                               flat_coordsX + p[0] * flat_coordsX + p[1] * flat_coordsY + p[2])

        # Compute error image
        errorImage = splineIt.ev(flat_coordsY, flat_coordsX) - warpedI

        # Compute weights using the chosen robust function
        weights = getWeightMatrix(errorImage, mtype=mtype)

        # Weighted Jacobian and Hessian
        SD = np.column_stack((Itx.flatten(), Ity.flatten())) @ J
        SD_weighted = SD.T * weights
        H = SD_weighted @ SD

        # Weighted steepest descent parameter update
        dp = np.linalg.pinv(H) @ (SD_weighted @ errorImage.flatten())

        # Update parameters
        p += dp.reshape(6, 1)

        # Break if change is below threshold
        if np.linalg.norm(dp) < threshold:
            break

    return p