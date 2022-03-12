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
rmt.start_tuning() # All

rmt.start_tuning(ACCELERATION + ROT_VELOCITY + MAGNETISM)
rmt.start_tuning(ACCELERATION + ROT_VELOCITY + TEMPERATURE)
rmt.start_tuning(ACCELERATION + MAGNETISM + TEMPERATURE)
rmt.start_tuning(ROT_VELOCITY + MAGNETISM + TEMPERATURE)

rmt.start_tuning(ACCELERATION + ROT_VELOCITY)
rmt.start_tuning(ACCELERATION + MAGNETISM)
rmt.start_tuning(ACCELERATION + TEMPERATURE)
rmt.start_tuning(ROT_VELOCITY + MAGNETISM)
rmt.start_tuning(ROT_VELOCITY + TEMPERATURE)
rmt.start_tuning(MAGNETISM + TEMPERATURE)

rmt.start_tuning(ACCELERATION)
rmt.start_tuning(ROT_VELOCITY)
rmt.start_tuning(MAGNETISM)
rmt.start_tuning(TEMPERATURE)
