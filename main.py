import numpy as np
from RegressionModelTrainer import RegressionModelTrainer as RMTrainer

DATA_PATH       = "TrainingData.csv"
LABEL_NAME      = "Viscosity"

ACCELERATION    = ["Xa", "Ya", "Za"]
ROT_VELOCITY    = ["Xg", "Yg", "Zg"]
MAGNETISM       = ["Xm", "Ym", "Zm"]

# Make NumPy printouts easier to read.
np.set_printoptions(suppress=True)

rmt = RMTrainer(DATA_PATH, LABEL_NAME)

# Searches with every combination of data type
rmt.start_tuning() # All

rmt.start_tuning(ACCELERATION + ROT_VELOCITY)
rmt.start_tuning(ACCELERATION + MAGNETISM)
rmt.start_tuning(ROT_VELOCITY + MAGNETISM)

rmt.start_tuning(ACCELERATION)
rmt.start_tuning(ROT_VELOCITY)
rmt.start_tuning(MAGNETISM)