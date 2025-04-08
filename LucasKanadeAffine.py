import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanadeAffine(It, It1, rect):
    """
    Lucas-Kanade Forward Additive Alignment with Affine Transformation
    Parameters:
        It: Template image (previous frame)
        It1: Current image (current frame)
        rect: Current position of the object (x1, y1, x2, y2)
    Returns:
        M: The 2x3 affine transformation matrix
    """
    # Threshold and maximum iterations
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64

    # Initialize affine warp parameters
    p = np.zeros(6, dtype=npDtype)  # [p1, p2, p3, p4, p5, p6]

    # Rectangle coordinates
    x1, y1, x2, y2 = rect

    # Dimensions of the image
    height, width = It.shape
    _x, _y = np.arange(width), np.arange(height)

    # Create splines for the images
    splineT = RectBivariateSpline(_x, _y, It.T)
    splineI = RectBivariateSpline(_x, _y, It1.T)

    # Generate coordinates for the template
    coordsX = np.linspace(x1, x2, int(x2 - x1), dtype=npDtype)
    coordsY = np.linspace(y1, y2, int(y2 - y1), dtype=npDtype)
    xx, yy = np.meshgrid(coordsX, coordsY)
    template = splineT.ev(xx, yy)

    for _ in range(maxIters):
        # Warp coordinates using the current affine parameters
        warped_x = p[0] * xx + p[1] * yy + p[2] + xx
        warped_y = p[3] * xx + p[4] * yy + p[5] + yy

        # Evaluate the warped image
        warped_image = splineI.ev(warped_x, warped_y)

        # Compute the error image
        error = template - warped_image

        # Gradients of the warped image
        grad_x = splineI.ev(warped_x, warped_y, dx=1)
        grad_y = splineI.ev(warped_x, warped_y, dy=1)

        # Build the steepest descent images
        n_pixels = xx.size
        A = np.zeros((n_pixels, 6))
        A[:, 0] = grad_x.ravel() * xx.ravel()
        A[:, 1] = grad_x.ravel() * yy.ravel()
        A[:, 2] = grad_x.ravel()
        A[:, 3] = grad_y.ravel() * xx.ravel()
        A[:, 4] = grad_y.ravel() * yy.ravel()
        A[:, 5] = grad_y.ravel()

        # Compute the Hessian matrix
        H = A.T @ A

        # Compute delta p
        b = error.ravel()
        dp = np.linalg.inv(H) @ (A.T @ b)

        # Update the affine parameters
        p += dp

        # Termination condition
        if np.linalg.norm(dp) < threshold:
            break

    # Reshape the output affine matrix
    M = np.array([[1.0 + p[0], p[1], p[2]],
                  [p[3], 1.0 + p[4], p[5]]])

    return M
