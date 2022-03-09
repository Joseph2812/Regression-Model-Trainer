import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

class RegressionModelTrainer:
        """
        Trains on a given dataset, and tries to find the optimal model by minimising validation loss.\n
        Use start_search() to find the best model with your desired features.
        """

        # Current Directory: MODELS_NAME\SESSION_NAME\CHECKPOINT_FILENAME
        MODELS_NAME             = "BestModels"
        SESSION_NAME            = "TrainingSession_{count:d}"
        CHECKPOINT_FILENAME     = "model_E{epoch:02d}-VL{val_loss:f}"

        RESULTS_FILENAME        = "Results.txt"
        RESULTS_CONTENT         = "\nTraining Session {count:d}: Epoch={epoch:d}, Loss={loss:f}, Validation_Loss={val_loss:f}"

        PLOTS_DIRECTORY         = "ModelPlots"
        PLOT_LOSS               = True

        # 1 iteration ~ max_epochs * (math.log(max_epochs, factor) ** 2) runs
        MAX_EPOCHS              = 100
        FACTOR                  = 3
        HYPERBAND_ITERATIONS    = 3
        PATIENCE                = 5
        
        def __init__(self, data_path:str, label_name:str):
                """Initialises the trainer with a dataset.

                Args:
                        data_path (str): Path to the dataset, relative to this program's location.\n
                        label_name (str): Name of the label to train towards, should match the column name in the dataset.
                """

                # Tracks how many times start_training() is run, so it can name the folder accordingly
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

                # Setting up directories and files
                if os.path.exists(self.MODELS_NAME):
                        shutil.rmtree(self.MODELS_NAME) # Clear previous models
                os.mkdir(self.MODELS_NAME)

                if os.path.exists(self.PLOTS_DIRECTORY):
                        shutil.rmtree(self.PLOTS_DIRECTORY) # Clear previous plots
                if self.PLOT_LOSS: os.mkdir(self.PLOTS_DIRECTORY)

                with open(self.RESULTS_FILENAME, 'w') as f:
                        f.write("=====Best Models=====") # Creates txt file or clears an existing one

                self.change_dataset(data_path, label_name)
        
        def change_dataset(self, data_path:str, label_name:str):
                """Loads a new dataset.

                Args:
                        data_path (str): Path to the dataset, relative to this program's location.\n
                        label_name (str): Name of the label to train towards, should match the column name in the dataset.
                """
                
                dataset = pd.read_csv(
                        data_path,
                        header=0,
                )

                # Split data into training and validation segments
                self.__data["Train"]["Features"] = dataset.sample(frac=0.8, random_state=0)
                self.__data["Valid"]["Features"] = dataset.drop(self.__data["Train"]["Features"].index)

                # Split the output (target variable) into its own labels variable
                self.__data["Train"]["Labels"] = self.__data["Train"]["Features"].pop(label_name)
                self.__data["Valid"]["Labels"] = self.__data["Valid"]["Features"].pop(label_name)

        def start_search(self, selected_columns:str=[]):
                """Start trialing various models, and fully train the best model found.
                
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

                self.__model_tuner()

                self.__training_count += 1

        def __model_tuner(self):
                tuner = kt.Hyperband(
                        self.__compile_model,
                        objective="val_loss",
                        max_epochs=self.MAX_EPOCHS,
                        factor=self.FACTOR,
                        hyperband_iterations=self.HYPERBAND_ITERATIONS,
                        directory="Trials",
                        project_name=self.SESSION_NAME.format(count=self.__training_count)
                )
                tuner.search(
                        self.__selected_train_features,
                        self.__data["Train"]["Labels"],
                        epochs=self.MAX_EPOCHS,
                        validation_data=(self.__selected_valid_features, self.__data["Valid"]["Labels"]),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.PATIENCE, restore_best_weights=True)]
                )

                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                
                # Build the best model (with best hyperparameters) and train it for MAX_EPOCHS
                model = tuner.hypermodel.build(best_hps)
                print("\nTraining the model with the best hyperparameters...")
                history = self.__train_model(model)

                print("\n=====Best Model=====")

                # Print out which has the lowest validation loss at the best epoch
                val_loss_per_epoch = history.history["val_loss"]               
                best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
                print("\nBest epoch: {:d}".format(best_epoch))
                print("Lowest validation loss: {:f}".format(min(val_loss_per_epoch)))               

                # Preview predictions of the new model
                new_model = tf.keras.models.load_model(self.__current_dir.format(epoch=best_epoch, val_loss=min(val_loss_per_epoch)))
                print("\nPredicting with 5 validation features to preview the label:\n\n---Features---")
                print(self.__selected_valid_features.head())

                label_name = self.__data["Valid"]["Labels"].name
                print("\n---Predicted Labels ({})---".format(label_name))
                print(new_model.predict(self.__selected_valid_features)[:5].transpose())
                print("\n---Validation Labels ({})---".format(label_name))
                print(self.__data["Valid"]["Labels"].head(), end="\n\n")

                # Save results
                with open(self.RESULTS_FILENAME, 'a') as f:
                        f.write(self.RESULTS_CONTENT.format(
                                count=self.__training_count,
                                epoch=best_epoch,
                                loss=min(history.history["loss"]),
                                val_loss=min(val_loss_per_epoch)
                        ))

        def __compile_model(self, hp):
                # Normalise the features
                normalizer = tf.keras.layers.Normalization(axis=-1)
                normalizer.adapt(np.array(self.__selected_train_features))
                
                # Create model
                model = tf.keras.Sequential([normalizer])

                for i in range(hp.Int("layers", 0, 5)): # 0 layers will make the model linear
                       model.add(tf.keras.layers.Dense(
                               units=hp.Int("units_" + str(i), 32, 256, step=32),
                               activation="relu"
                        ))
                model.add(tf.keras.layers.Dense(1))
               
                model.summary()

                hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
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
                        plt.ylabel("Error")
                        plt.legend()
                        plt.grid(True)
                        fig.savefig(self.PLOTS_DIRECTORY + '\\' + title + ".png")

                return history
                