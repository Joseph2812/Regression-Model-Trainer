import os
import shutil
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error
from Classes.regression_model_trainer import RegressionModelTrainer as RMTrainer

class RegressionModelTrainerTensorFlow(RMTrainer):
    RESULTS_CONTENT  = "<TensorFlow>: Hyperparameters: Layers={layers:d}, Learning_Rate={rate:.0e}, Units={units}."
    TRIALS_DIRECTORY = "TensorFlow Hyperparameter Trials"

    ACTIVATION_FUNCTION = "relu"

    parameters:dict[str, any] = {
        "objective": "mean_squared_error", # Objective == Evaluation metric

        # Hyperband parameters
        # 1 iteration ~ max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs
        "max_epochs"          : 100,
        "factor"              : 3,
        "hyperband_iterations": 1,
        "patience"            : 5,

        # Hyperparameter tuning
        "min_hidden_layers": 0,
        "max_hidden_layers": 10,

        "min_units": 128,
        "max_units": 1024,
        "unit_step": 128,

        "learning_rate_choice": [1e-3, 1e-4, 1e-5]
    }

    def __init__(self, keep_previous_trials:bool=True):
        """Initialises the tensorflow trainer, and choose whether to resume from previous trials.

        Args:
            keep_previous_trials (bool): Whether you would like the tuner to reuse old trials, good for resuming search on the same dataset & tuning parameters. Default = True.
        """
        
        super().__init__()
        self._set_trainer_name("TensorFlow")

        if not keep_previous_trials:
            if os.path.exists(self.TRIALS_DIRECTORY):
                shutil.rmtree(self.TRIALS_DIRECTORY) # Clear previous trials

    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        # Search through various hyperparameters to see which model gives the lowest validation loss
        tuner = kt.Hyperband(
            self.__compile_model,
            objective="val_loss",
            max_epochs=self.parameters["max_epochs"],
            factor=self.parameters["factor"],
            hyperband_iterations=self.parameters["hyperband_iterations"],
            directory=self.TRIALS_DIRECTORY,
            project_name=self.SESSION_NAME.format(count=self._training_count)
        )
        tuner.search(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.parameters["max_epochs"],
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.parameters["patience"], restore_best_weights=True)]
        )
        best_hps = tuner.get_best_hyperparameters()[0]
        
        # Build the best model (with best hyperparameters) and train it for MAX_EPOCHS
        model = tuner.hypermodel.build(best_hps)

        print("\n[TensorFlow] Training the model with the best hyperparameters...")
        history = self.__train_model(model)      
        val_losses = history.history["val_loss"]
        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(val_losses)

        # Get predictions of the new model
        model = tf.keras.models.load_model(self._model_dir.format(epoch=best_epoch, val_loss=lowest_val_loss))
        predictions = model.predict(self._selected_train_features)
        test_predictions = model.predict(self._selected_test_features)
        
        # Saving best model data
        units_list = []
        for i in range(best_hps["layers"]):
            units_list.append(best_hps["units_{:d}".format(i)])

        self._save_results(
            epoch=best_epoch,
            loss=min(history.history["loss"]),
            val_loss=lowest_val_loss,
            test_loss=mean_absolute_error(self._data["test"]["labels"], test_predictions),
            unique_results=self.RESULTS_CONTENT.format(
                layers=best_hps["layers"],
                rate=best_hps["learning_rate"],
                units=units_list
            )       
        )

        return (self.parameters["objective"], history.history["loss"], val_losses, predictions)

    def __train_model(self, model) -> any:
        history = model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.parameters["max_epochs"],
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[tf.keras.callbacks.ModelCheckpoint(
                self._model_dir,
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
        normaliser.adapt(self._selected_train_features.to_numpy())
        
        # ---Create model---
        model = tf.keras.Sequential([normaliser])

        # 0 hidden layers will make the model linear
        for i in range(hp.Int("layers", self.parameters["min_hidden_layers"], self.parameters["max_hidden_layers"])): 
            model.add(
                tf.keras.layers.Dense(
                    units=hp.Int("units_" + str(i), self.parameters["min_units"], self.parameters["max_units"], step=self.parameters["unit_step"]),
                    activation=self.ACTIVATION_FUNCTION
                )
            )
        model.add(tf.keras.layers.Dense(1))
        
        model.summary()

        hp_learning_rate = hp.Choice("learning_rate", values=self.parameters["learning_rate_choice"])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=self.parameters["objective"]
        )
        
        return model