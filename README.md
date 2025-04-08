This project implements and compares two classical object tracking algorithms—Lucas-Kanade Tracker and Matthews-Baker Inverse Compositional Tracker—on various video sequences. The goal is to analyze tracking accuracy, efficiency, and robustness under challenging conditions like occlusion and lighting changes.

🎯 Objectives


✅ Implement Lucas-Kanade Tracker using both Translation and Affine Transformation models.

✅ Develop the Matthews-Baker Inverse Compositional Tracker for enhanced computational efficiency.

✅ Compare Forward Additive and Inverse Compositional methods in terms of performance.

✅ Test and visualize results on real-world video sequences (e.g., car tracking, landing sequences).

⚙️ Methodology


🧱 Use image pyramids and interpolation to handle large object displacements.

🧮 Compute image gradients and Jacobians for warp parameter estimation.

➗ Solve for motion parameters via least squares optimization.

🛑 Implement a termination condition based on error thresholds to optimize computation.

🧰 Tools & Technologies


Programming Language: Python

Libraries: OpenCV, NumPy, SciPy, Matplotlib

Testing Framework: Provided video datasets and tracking scripts

🚀 Expected Outcome


✅ A robust tracking system that detects and follows a target object across frames.

📊 Comparative analysis of tracking accuracy and efficiency between Lucas-Kanade and Matthews-Baker methods.

🧪 Insight into tracker limitations under occlusion, illumination changes, and motion blur.
