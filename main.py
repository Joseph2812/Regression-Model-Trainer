import numpy as np
from RegressionModelClasses.regression_model_trainer_tensorflow import RegressionModelTrainerTensorFlow as RMTrainerTF
from RegressionModelClasses.regression_model_trainer_xgboost import RegressionModelTrainerXGBoost as RMTrainerXGB

DATA_PATH       = "TrainingData.csv"
LABEL_NAME      = "Viscosity"

ACCELERATION    = ["Xa", "Ya", "Za"]
ROT_VELOCITY    = ["Xg", "Yg", "Zg"]
MAGNETISM       = ["Xm", "Ym", "Zm"]

# Make NumPy printouts easier to read.
np.set_printoptions(suppress=True)

RMTrainerXGB.reset_files()
rmt_tf = RMTrainerXGB(DATA_PATH, LABEL_NAME)

# ---Searches every combination of the data---
rmt_tf.start_training() # All

#rmt_tf.start_training(ACCELERATION + ROT_VELOCITY)
#rmt_tf.start_training(ACCELERATION + MAGNETISM)
#rmt_tf.start_training(ROT_VELOCITY + MAGNETISM)

#rmt_tf.start_training(ACCELERATION)
#rmt_tf.start_training(ROT_VELOCITY)
#rmt_tf.start_training(MAGNETISM)