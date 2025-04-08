import numpy as np
from scipy.interpolate import RectBivariateSpline

def InverseCompositionAffine(It, It1, rect):
    """
    Matthews-Baker Inverse Compositional Alignment with Affine Transformation
    Parameters:
        It: Template image (previous frame)
        It1: Current image (current frame)
        rect: Current position of the object (x1, y1, x2, y2)
    Returns:
        M: The 2x3 affine transformation matrix
    """
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64

    W = np.eye(3, dtype=npDtype)[:2, :]  # Initial warp matrix (2x3)
    x1, y1, x2, y2 = rect

    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    splineT = RectBivariateSpline(_x, _y, It.T)
    
    coordsX = np.linspace(x1, x2, int(x2 - x1), dtype=npDtype)
    coordsY = np.linspace(y1, y2, int(y2 - y1), dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)
    template = splineT.ev(xx, yy)

    grad_x = splineT.ev(xx, yy, dx=1)
    grad_y = splineT.ev(xx, yy, dy=1)

    A = np.zeros((len(grad_x.ravel()), 6))
    A[:, 0] = grad_x.ravel() * xx.ravel()
    A[:, 1] = grad_x.ravel() * yy.ravel()
    A[:, 2] = grad_x.ravel()
    A[:, 3] = grad_y.ravel() * xx.ravel()
    A[:, 4] = grad_y.ravel() * yy.ravel()
    A[:, 5] = grad_y.ravel()

    H = A.T @ A
    splineI = RectBivariateSpline(_x, _y, It1.T)

    for _ in range(maxIters):
        warped_x = W[0, 0] * xx + W[0, 1] * yy + W[0, 2]
        warped_y = W[1, 0] * xx + W[1, 1] * yy + W[1, 2]

        warped_image = splineI.ev(warped_x, warped_y)
        error = template - warped_image

        b = A.T @ error.ravel()
        dp = np.linalg.inv(H) @ b

        dp_mat = np.array([[1 + dp[0], dp[1], dp[2]],
                           [dp[3], 1 + dp[4], dp[5]],
                           [0, 0, 1]])

        W_full = np.vstack([W, [0, 0, 1]])
        W_full = np.linalg.inv(dp_mat) @ W_full
        W = W_full[:2, :]

        if np.linalg.norm(dp) < threshold:
            break

    M = W[:2, :]
    return M
