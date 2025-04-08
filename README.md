3️⃣ Objectives
Implement Lucas-Kanade Tracker (Translation & Affine Transformation).

Implement Matthews-Baker Inverse Compositional Tracker for efficiency.

Compare the performance of Forward Additive vs. Inverse Compositional Methods.

Test and analyze results on video sequences (car tracking, landing sequence, etc.).

4️⃣ Methodology
Use image pyramids and interpolation for handling large movements.

Compute image gradients and Jacobians for warp estimation.

Solve for motion parameters using least squares optimization.

Implement a termination condition to optimize computation.

5️⃣ Tools & Technologies
Programming Language: Python

Libraries: OpenCV, NumPy, SciPy, Matplotlib

Testing Framework: Provided video datasets & tracking scripts

6️⃣ Expected Outcome
A working object tracking system that can detect and follow an object in a video.

Comparison of tracking accuracy and computational efficiency between Lucas-Kanade and Matthews-Baker methods.

Analyzed tracking failures under challenging conditions (e.g., occlusion, illumination changes).
