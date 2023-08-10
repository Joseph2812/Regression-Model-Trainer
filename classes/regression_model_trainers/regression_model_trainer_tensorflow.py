import os
import shutil
import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import mean_absolute_error
from classes.live_plotter.plotter_process_manager import PlotterProcessManager
from classes.regression_model_trainers.regression_model_trainer import RegressionModelTrainer as RMTrainer

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
        tuner = kt.Hyperband(
            self.__compile_model,
            objective="val_loss",
            max_epochs=self.parameters["max_epochs"],
            factor=self.parameters["factor"],
            hyperband_iterations=self.parameters["hyperband_iterations"],
            directory=self.TRIALS_DIRECTORY,
            project_name=self.SESSION_NAME.format(count=self._training_count)
        )

        plot_manager = PlotterProcessManager("Validation Loss at Current Hyperparameters", "Epoch", self.parameters["objective"])

        # Search through various hyperparameters to see which model gives the lowest validation loss
        tuner.search(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.parameters["max_epochs"],
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.parameters["patience"], restore_best_weights=True),
                tf.keras.callbacks.LambdaCallback(
                    on_train_begin=lambda _: plot_manager.plot(None, None)
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: plot_manager.plot(epoch + 1, logs["val_loss"])
                )
            ]
        ) 
       
        # Build the best model (with best hyperparameters) and train it for MAX_EPOCHS
        best_hps = tuner.get_best_hyperparameters()[0]
        model = tuner.hypermodel.build(best_hps)

        print("\n[TensorFlow] Training the model with the best hyperparameters...")
        history = self.__train_model(model, plot_manager)
        plot_manager.end_process()

        val_losses = history.history["val_loss"]
        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(val_losses)

        # Get predictions of the new model
        model = tf.keras.models.load_model(self._model_dir.format(epoch=best_epoch, val_loss=lowest_val_loss))
        predictions = model.predict(self._all_selected_features)
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

    def __train_model(self, model, plot_manager) -> any:
        history = model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            epochs=self.parameters["max_epochs"],
            validation_data=(self._selected_valid_features, self._data["valid"]["labels"]),
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(
                    self._model_dir,
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="min",
                    save_freq="epoch"
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_train_begin=lambda _: plot_manager.plot(None, None)
                ),
                tf.keras.callbacks.LambdaCallback(
                    on_epoch_end=lambda epoch, logs: plot_manager.plot(epoch + 1, logs["val_loss"])
                )
            ]
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