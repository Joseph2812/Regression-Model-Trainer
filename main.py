import numpy as np
import tensorflow as tf
from RegressionModelTrainer import RegressionModelTrainer as RMTrainer

DATA_PATH       = "TrainingData.csv"
LABEL_NAME      = "Viscosity"

ACCELERATION    = ["Xa", "Ya", "Za"]
ROT_VELOCITY    = ["Xg", "Yg", "Zg"]
MAGNETISM       = ["Xm", "Ym", "Zm"]
TEMPERATURE     = ["T"]

# Make NumPy printouts easier to read.
np.set_printoptions(suppress=True)

rmt = RMTrainer(DATA_PATH, LABEL_NAME)

# Searches with every combination of data type
rmt.start_search() # All

rmt.start_search(ACCELERATION + ROT_VELOCITY + MAGNETISM)
rmt.start_search(ACCELERATION + ROT_VELOCITY + TEMPERATURE)
rmt.start_search(ACCELERATION + MAGNETISM + TEMPERATURE)
rmt.start_search(ROT_VELOCITY + MAGNETISM + TEMPERATURE)

rmt.start_search(ACCELERATION + ROT_VELOCITY)
rmt.start_search(ACCELERATION + MAGNETISM)
rmt.start_search(ACCELERATION + TEMPERATURE)
rmt.start_search(ROT_VELOCITY + MAGNETISM)
rmt.start_search(ROT_VELOCITY + TEMPERATURE)
rmt.start_search(MAGNETISM + TEMPERATURE)

rmt.start_search(ACCELERATION)
rmt.start_search(ROT_VELOCITY)
rmt.start_search(MAGNETISM)
rmt.start_search(TEMPERATURE)
