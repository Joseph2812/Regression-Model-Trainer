import numpy as np
import tensorflow as tf
from NonLinearRegressionModelTrainer import NonLinearRegressionModelTrainer as nlrmTrainer

# Make NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
print(tf.__version__)

mmt = nlrmTrainer()