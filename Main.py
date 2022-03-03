import numpy as np
import tensorflow as tf
from RegressionModelTrainer import RegressionModelTrainer as RMTrainer

DATA_PATH = "TrainingData.csv"
COLUMN_NAMES = ["Xg", "Yg", "Zg", "Xa", "Ya", "Za", "T", "Xm", "Ym", "Zm", "Viscosity"]

# Make NumPy printouts easier to read.
np.set_printoptions(precision=5, suppress=True)

rmt = RMTrainer(DATA_PATH, COLUMN_NAMES)