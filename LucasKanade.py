import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    """
    Lucas-Kanade Forward Additive Alignment with Translation
    Parameters:
        It: Template image (previous frame)
        It1: Current image (current frame)
        rect: Current position of the object (x1, y1, x2, y2)
    Returns:
        p: Movement vector (dx, dy)
    """
    # Threshold and maximum iterations
    threshold = 0.01875
    maxIters = 100
    npDtype = np.float64
    
    # Initialize motion vector [dx, dy]
    p = np.zeros(2, dtype=npDtype)  # dx, dy
    
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
        # Warp coordinates using the current motion estimate
        warped_x = xx + p[0]
        warped_y = yy + p[1]

        # Evaluate the current warped image
        warped_image = splineI.ev(warped_x, warped_y)

        # Compute the error image
        error = template - warped_image

        # Gradients of the warped image
        grad_x = splineI.ev(warped_x, warped_y, dx=1)
        grad_y = splineI.ev(warped_x, warped_y, dy=1)

        # Build the steepest descent images
        A = np.stack([grad_x.ravel(), grad_y.ravel()], axis=-1)

        # Compute the Hessian matrix
        H = A.T @ A

        # Compute delta p
        b = error.ravel()
        dp = np.linalg.inv(H) @ (A.T @ b)

        # Update the motion vector
        p += dp

        # Termination condition
        if np.linalg.norm(dp) < threshold:
            break

    return p
