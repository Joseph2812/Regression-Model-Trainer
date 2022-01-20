import numpy as np
import tensorflow as tf
from RegressionModelTrainer import RegressionModelTrainer as RMTrainer

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)

rmt = RMTrainer()