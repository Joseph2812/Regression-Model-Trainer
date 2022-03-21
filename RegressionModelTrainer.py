import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

class RegressionModelTrainer:
        """
        Trains on a given dataset, and tries to find the optimal model by minimising validation loss (MAE).\n
        Use start_tuning() to find the best model with your desired features.
        """

        # Current directory: MODELS_NAME\SESSION_NAME\CHECKPOINT_FILENAME
        MODELS_NAME             = "BestModels"
        SESSION_NAME            = "TrainingSession_{count:d}"
        CHECKPOINT_FILENAME     = "model_E{epoch:02d}-VL{val_loss:f}"

        TRIALS_DIRECTORY        = "Trials"

        PLOTS_DIRECTORY         = "ModelPlots"
        PLOT_LOSS               = True

        RESULTS_FILENAME        = "Results.txt"
        RESULTS_CONTENT         = "[Training Session {count:d}]: Epoch={epoch:d}, Loss={loss:f}, Validation_Loss={val_loss:f}. Hyperparameters: Layers={layers:d}, Units={units}, Learning_Rate={rate:.0e}."
        
        # Hyperband parameters
        # 1 iteration ~ max_epochs * (math.log(max_epochs, factor) ** 2) cumulative epochs
        MAX_EPOCHS              = 100
        FACTOR                  = 3
        HYPERBAND_ITERATIONS    = 2
        PATIENCE                = 5

        # Hyperparameter tuning
        MIN_HIDDEN_LAYERS       = 0
        MAX_HIDDEN_LAYERS       = 16

        UNIT_STEP               = 32
        MIN_UNITS               = UNIT_STEP
        MAX_UNITS               = 1024

        LEARNING_RATE_CHOICE    = [1e-2, 1e-3, 1e-4, 1e-5]
        
        def __init__(self, data_path:str, label_name:str, keep_previous_trials:bool=False):
                """Initialises the trainer with a dataset.

                Args:
                        data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
                        label_name (str): Name of the label to train towards, should match the column name in the dataset.\n
                        keep_previous_trials (bool): Whether you would like the tuner to reuse old trials, good for resuming search on the same dataset & tuning inputs. Default = False.
                """

                # Tracks how many times start_tuning() is run, so it can name the folder accordingly
                self.__training_count = 0
                self.__current_dir = ""

                # Dictionary structure to store the split up dataset
                self.__data = {
                        "Train": {"Features": [], "Labels": []},
                        "Valid": {"Features": [], "Labels": []}
                }

                # For feature selection
                self.__selected_train_features = []
                self.__selected_valid_features = []

                # ---Setting up directories and files---
                if os.path.exists(self.MODELS_NAME):
                        shutil.rmtree(self.MODELS_NAME) # Clear previous models
                os.mkdir(self.MODELS_NAME)

                if not keep_previous_trials:
                        if os.path.exists(self.TRIALS_DIRECTORY):
                                shutil.rmtree(self.TRIALS_DIRECTORY) # Clear previous trials
                        os.mkdir(self.TRIALS_DIRECTORY)              

                if os.path.exists(self.PLOTS_DIRECTORY):
                        shutil.rmtree(self.PLOTS_DIRECTORY) # Clear previous plots
                if self.PLOT_LOSS:
                        os.mkdir(self.PLOTS_DIRECTORY)

                with open(self.RESULTS_FILENAME, 'w') as f:
                        f.write("=====Best Models=====") # Creates txt file or clears an existing one

                # Load initial dataset
                self.change_dataset(data_path, label_name)
        
        def change_dataset(self, data_path:str, label_name:str):
                """Loads a new dataset.

                Args:
                        data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
                        label_name (str): Name of the label to train towards, should match the column name in the dataset.
                """
                
                dataset = pd.read_csv(data_path, header=0)

                # Split data into training and validation segments
                self.__data["Train"]["Features"] = dataset.sample(frac=0.8, random_state=0)
                self.__data["Valid"]["Features"] = dataset.drop(self.__data["Train"]["Features"].index)

                # Split the output (target variable) into its own labels variable
                self.__data["Train"]["Labels"] = self.__data["Train"]["Features"].pop(label_name)
                self.__data["Valid"]["Labels"] = self.__data["Valid"]["Features"].pop(label_name)

        def start_tuning(self, selected_columns:str=[]):
                """Start trialling various models, and fully train the best model found.
                
                Args:
                        selected_columns (str): Columns that you want to include in training, leave empty to use all of the columns. Default = [].
                """

                print("\n=====Training Session {:d}=====".format(self.__training_count))
                        
                # Update directory path for this training session
                self.__current_dir = self.MODELS_NAME + '\\' + self.SESSION_NAME.format(count=self.__training_count) + '\\' + self.CHECKPOINT_FILENAME

                # Select requested feature columns
                if selected_columns == []:
                        self.__selected_train_features = self.__data["Train"]["Features"]
                        self.__selected_valid_features = self.__data["Valid"]["Features"]
                else:
                        self.__selected_train_features = self.__data["Train"]["Features"][selected_columns]
                        self.__selected_valid_features = self.__data["Valid"]["Features"][selected_columns]

                # Show selected data (preview to see if it's setup right)
                print("\n---Selected Training Data---")
                print(self.__selected_train_features.head(), end="\n\n")
                print(self.__data["Train"]["Labels"].head())

                print("\n---Selected Validation Data---")
                print(self.__selected_valid_features.head(), end="\n\n")
                print(self.__data["Valid"]["Labels"].head(), end="\n\n")

                self.__tune_models()

                self.__training_count += 1

        def __tune_models(self):
                # Search through various hyperparameters to see which model gives the lowest validation loss
                tuner = kt.Hyperband(
                        self.__compile_model,
                        objective="val_loss",
                        max_epochs=self.MAX_EPOCHS,
                        factor=self.FACTOR,
                        hyperband_iterations=self.HYPERBAND_ITERATIONS,
                        directory=self.TRIALS_DIRECTORY,
                        project_name=self.SESSION_NAME.format(count=self.__training_count)
                )
                tuner.search(
                        self.__selected_train_features,
                        self.__data["Train"]["Labels"],
                        epochs=self.MAX_EPOCHS,
                        validation_data=(self.__selected_valid_features, self.__data["Valid"]["Labels"]),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.PATIENCE, restore_best_weights=True)]
                )
                best_hps = tuner.get_best_hyperparameters()[0]
                
                # Build the best model (with best hyperparameters) and train it for MAX_EPOCHS
                model = tuner.hypermodel.build(best_hps)

                print("\nTraining the model with the best hyperparameters...")
                history = self.__train_model(model)

                print("\n=====Best Model=====")

                # Print out which has the lowest validation loss at the best epoch
                val_loss_per_epoch = history.history["val_loss"]

                lowest_val_loss = min(val_loss_per_epoch)
                best_epoch = val_loss_per_epoch.index(lowest_val_loss) + 1

                print("\nBest epoch: {:d}".format(best_epoch))
                print("Lowest validation loss: {:f}".format(lowest_val_loss))               

                # Preview predictions of the new model
                new_model = tf.keras.models.load_model(self.__current_dir.format(epoch=best_epoch, val_loss=lowest_val_loss))

                print("\nPredicting with 5 validation features to preview the label:\n\n---Validation Features---")
                print(self.__selected_valid_features.head())

                label_name = self.__data["Valid"]["Labels"].name

                print("\n---Predicted Labels ({})---".format(label_name))
                print(new_model.predict(self.__selected_valid_features)[:5].transpose())

                print("\n---Validation Labels ({})---".format(label_name))
                print(self.__data["Valid"]["Labels"].head(), end="\n\n")

                # Formatting unit values for printing
                layers = best_hps["layers"]
                units = '['
                for i in range(layers):
                        units += str(best_hps["units_{:d}".format(i)])
                        if i != (layers - 1): units += ", "
                units += ']'

                # Save results
                with open(self.RESULTS_FILENAME, 'a') as f:                       
                        f.write('\n' + self.RESULTS_CONTENT.format(
                                count=self.__training_count,
                                epoch=best_epoch,
                                loss=min(history.history["loss"]),
                                val_loss=lowest_val_loss,

                                layers=layers,
                                units=units,
                                rate=best_hps["learning_rate"]
                        ))

        def __compile_model(self, hp):
                # Normalise the features
                normalizer = tf.keras.layers.Normalization()
                normalizer.adapt(np.array(self.__selected_train_features))
                
                # Create model
                model = tf.keras.Sequential([normalizer])

                for i in range(hp.Int("layers", self.MIN_HIDDEN_LAYERS, self.MAX_HIDDEN_LAYERS)): # 0 layers will make the model linear
                       model.add(tf.keras.layers.Dense(
                               units=hp.Int("units_" + str(i), self.MIN_UNITS, self.MAX_UNITS, step=self.UNIT_STEP),
                               activation="relu"
                        ))
                model.add(tf.keras.layers.Dense(1))
               
                model.summary()

                hp_learning_rate = hp.Choice("learning_rate", values=self.LEARNING_RATE_CHOICE)
                model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss="mean_absolute_error"
                )
                return model

        def __train_model(self, model):               
                history = model.fit(
                        self.__selected_train_features,
                        self.__data["Train"]["Labels"],
                        epochs=self.MAX_EPOCHS,
                        validation_data=(self.__selected_valid_features, self.__data["Valid"]["Labels"]),
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(
                                self.__current_dir,
                                monitor="val_loss",
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=False,
                                mode="min",
                                save_freq="epoch"
                        )]
                )

                if self.PLOT_LOSS:                      
                        title = self.SESSION_NAME.format(count=self.__training_count)

                        # Plots loss and val_loss
                        fig = plt.figure()
                        plt.plot(history.history["loss"], label="loss")
                        plt.plot(history.history["val_loss"], label="val_loss")
                        plt.ylim([0, max(history.history["loss"]) * 1.1]) # 10% padding
                        plt.title(title)
                        plt.xlabel("Epoch")
                        plt.ylabel("Mean Absolute Error")
                        plt.legend()
                        plt.grid(True)
                        fig.savefig(self.PLOTS_DIRECTORY + '\\' + title + ".png")

                return history
                