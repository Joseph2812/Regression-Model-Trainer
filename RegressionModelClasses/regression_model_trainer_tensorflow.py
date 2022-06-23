import os
import shutil
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from RegressionModelClasses.regression_model_trainer import RegressionModelTrainer as RMTrainer

class RegressionModelTrainerTensorFlow(RMTrainer):
    RESULTS_CONTENT     = "<TensorFlow>: Epoch={epoch:d}, Loss={loss:f}, Validation_Loss={val_loss:f}. Hyperparameters: Layers={layers:d}, Learning_Rate={rate:.0e}, Units={units}."
    TRIALS_DIRECTORY    = "Trials"

    OBJECTIVE = "mean_squared_error" # Objective == Evaluation metric
    # Hyperband parameters
    # 1 iteration ~ max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs
    MAX_EPOCHS           = 10
    FACTOR               = 3
    HYPERBAND_ITERATIONS = 1
    PATIENCE             = 5

    # Hyperparameter tuning
    MIN_HIDDEN_LAYERS    = 9
    MAX_HIDDEN_LAYERS    = 10

    UNIT_STEP            = 256
    MIN_UNITS            = UNIT_STEP
    MAX_UNITS            = 1024

    LEARNING_RATE_CHOICE = [1e-4, 1e-5]

    def __init__(self, data_path:str, label_name:str, keep_previous_trials:bool=False):
        """Initialises the trainer with a dataset.

        Args:
            data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
            label_name (str): Name of the label to train towards, should match the column name in the dataset.\n
            keep_previous_trials (bool): Whether you would like the tuner to reuse old trials, good for resuming search on the same dataset & tuning parameters. Default = False.
        """
        
        super().__init__(data_path, label_name)
        self._set_trainer_name("TensorFlow")

        if not keep_previous_trials:
            if os.path.exists(self.TRIALS_DIRECTORY):
                shutil.rmtree(self.TRIALS_DIRECTORY) # Clear previous trials
            os.mkdir(self.TRIALS_DIRECTORY)

    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        # Search through various hyperparameters to see which model gives the lowest validation loss
        tuner = kt.Hyperband(
            self.__compile_model,
            objective="val_loss",
            max_epochs=self.MAX_EPOCHS,
            factor=self.FACTOR,
            hyperband_iterations=self.HYPERBAND_ITERATIONS,
            directory=self.TRIALS_DIRECTORY,
            project_name=self.SESSION_NAME.format(count=self._training_count)
        )
        tuner.search(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.MAX_EPOCHS,
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.PATIENCE, restore_best_weights=True)]
        )
        best_hps = tuner.get_best_hyperparameters()[0]
        
        # Build the best model (with best hyperparameters) and train it for MAX_EPOCHS
        model = tuner.hypermodel.build(best_hps)

        print("\n[TensorFlow] Training the model with the best hyperparameters...")
        history = self.__train_model(model)      
        val_losses = history.history["val_loss"]
        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(val_losses)

        # Get predictions of the new model
        new_model = tf.keras.models.load_model(self._current_dir.format(epoch=best_epoch, val_loss=lowest_val_loss))
        predictions = new_model.predict(self._selected_valid_features)[:5].transpose()
        
        # Saving best model data
        units_list = []
        for i in range(best_hps["layers"]):
            units_list.append(best_hps["units_{:d}".format(i)])

        self._save_results(self.RESULTS_CONTENT.format(
            epoch=best_epoch,
            loss=min(history.history["loss"]),
            val_loss=lowest_val_loss,
            layers=best_hps["layers"],
            learning_rate=best_hps["learning_rate"],
            units_list=units_list
        ))

        # Formatting unit values for saving
        #units_to_string = '['
        #for i, units in enumerate(units_list):
        #    units_to_string += str(units)
        #    if i != (layers - 1): units_to_string += ", "
        #units_to_string += ']'

        return (self.OBJECTIVE, history.history["loss"], val_losses, predictions)

    def __train_model(self, model) -> any:
        history = model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.MAX_EPOCHS,
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(
                self._current_dir,
                monitor="val_loss",
                verbose=0,
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                save_freq="epoch"
            )]
        )

        return history

    def __compile_model(self, hp) -> any:
        # Normalise the features
        normaliser = tf.keras.layers.Normalization()
        normaliser.adapt(np.array(self._selected_train_features))
        
        # ---Create model---
        model = tf.keras.Sequential([normaliser])

        # 0 hidden layers will make the model linear
        for i in range(hp.Int("layers", self.MIN_HIDDEN_LAYERS, self.MAX_HIDDEN_LAYERS)): 
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int("units_" + str(i), self.MIN_UNITS, self.MAX_UNITS, step=self.UNIT_STEP),
                    activation="relu"
                )
            )
        model.add(tf.keras.layers.Dense(1))
        
        model.summary()

        hp_learning_rate = hp.Choice("learning_rate", values=self.LEARNING_RATE_CHOICE)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.OBJECTIVE
        )
        return model