import os
import shutil
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

class RegressionModelTrainer:
        CHECKPOINT_DIR = "BestModels\\best_model_E{epoch:02d}-VL{val_loss:f}"

        def __init__(self, data_path, column_names):
                # For loading data
                self.__data_path = data_path
                self.__column_names = column_names

                # For training
                self.__train_features = [] # Input
                self.__train_labels = [] # Output

                # For validation
                self.__valid_features = [] # Input
                self.__valid_labels = [] # Output

                if os.path.exists("BestModels"):
                        shutil.rmtree("BestModels") # Clear previous models

                self.__setup_data()
                self.__model_tuner()
        
        def __setup_data(self):
                raw_dataset = pd.read_csv(
                        self.__data_path, 
                        names=self.__column_names,
                        header=0,
                )
                dataset = raw_dataset.copy()

                # Split data into training and validation segments
                train_dataset = dataset.sample(frac=0.8, random_state=0)
                valid_dataset = dataset.drop(train_dataset.index)

                # Copy the data to a new set of variables
                self.__train_features = train_dataset.copy()
                self.__valid_features = valid_dataset.copy()

                # Split the output (target variable) into its own label variable
                self.__train_labels = self.__train_features.pop("Viscosity")
                self.__valid_labels = self.__valid_features.pop("Viscosity")

                # Show training data (preview to see if it's setup right)
                print("---Training Data---")
                print(self.__train_features)
                print(self.__train_labels)

        def __compile_model(self, hp):
                # Normalise the features
                normalizer = tf.keras.layers.Normalization(axis=-1)
                normalizer.adapt(np.array(self.__train_features))
                
                # Create model
                model = tf.keras.Sequential([normalizer])

                for i in range(hp.Int("layers", 0, 5)): # 0 layers will make the model linear
                       model.add(tf.keras.layers.Dense(
                               units=hp.Int("units_" + str(i), 32, 256, step=32),
                               activation="relu"
                        ))
                model.add(tf.keras.layers.Dense(1))
               
                model.summary()

                # Choose an optimal value for learning rate
                hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
                model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                        loss="mean_absolute_error"
                )
                return model

        def __model_tuner(self):
                tuner = kt.Hyperband(
                        self.__compile_model,
                        objective="val_loss",
                        max_epochs=50,
                        factor=3,
                        directory="Trials",
                        project_name="DNN Models"
                )
                print(tuner.search_space_summary())

                tuner.search(
                        self.__train_features,
                        self.__train_labels,
                        epochs=100,
                        validation_data=(self.__valid_features, self.__valid_labels),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
                )

                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                
                # Build the best model (with best hyperparameters) and train it for a selected number of epochs (default = 100)
                model = tuner.hypermodel.build(best_hps)
                history = self.__train_model(model)

                # Print out which has the lowest validation loss at the best epoch
                val_loss_per_epoch = history.history["val_loss"]               
                best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
                print("Lowest validation loss: {:f}".format(min(val_loss_per_epoch)))
                print("Best epoch: {:d}".format(best_epoch))

                # Preview predictions of the new model
                new_model = tf.keras.models.load_model(self.CHECKPOINT_DIR.format(epoch=best_epoch, val_loss=min(val_loss_per_epoch)))
                print("Putting 5 validation inputs to preview the output\n---Input---")
                print(self.__valid_features[:5])
                print("---Output---")
                print(new_model.predict(self.__valid_features[:5]).transpose())


        def __train_model(self, model, epochs=100):
                history = model.fit(
                        self.__train_features,
                        self.__train_labels,
                        epochs=epochs,
                        validation_data=(self.__valid_features, self.__valid_labels),
                        callbacks=[tf.keras.callbacks.ModelCheckpoint(
                                self.CHECKPOINT_DIR,
                                monitor="val_loss",
                                verbose=0,
                                save_best_only=True,
                                save_weights_only=False,
                                mode="min",
                                save_freq="epoch"
                        )],
                        verbose=0 # Suppress logging
                )
                # Plot loss and val_loss
                plt.plot(history.history["loss"], label="loss")
                plt.plot(history.history["val_loss"], label="val_loss")
                plt.ylim([0, 5])
                plt.xlabel("Epoch")
                plt.ylabel("Error")
                plt.legend()
                plt.grid(True)
                plt.show()

                return history
                