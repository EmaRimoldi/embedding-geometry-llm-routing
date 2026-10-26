"""Fixed settings shared by the probes (see PLAN.md / PROTOCOL.md before changing any)."""
import numpy as np

RIDGE_ALPHAS = np.logspace(-1, 5, 13)
KNN_KS = (5, 20, 50, 100)
KMEANS_KS = (7, 64, 256)   # 7 = number of task families (PI, 2026-09-13)
LAMBDA_GRID = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
