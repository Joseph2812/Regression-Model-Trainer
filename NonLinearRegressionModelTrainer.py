import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras_tuner as kt

class NonLinearRegressionModelTrainer:
        def __init__(self):
                # For training
                self.__train_features = [] # Input
                self.__train_labels = [] # Output

                # For validation
                self.__valid_features = [] # Input
                self.__valid_labels = [] # Output

                self.__setup_data()
                self.__model_tuner()
        
        def __setup_data(self):
                url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
                column_names = ["MPG", "Cylinders", "Displacement", "Horsepower", "Weight", "Acceleration", "Model Year", "Origin"]

                raw_dataset = pd.read_csv(
                        url, 
                        names=column_names,
                        na_values="?",
                        comment="\t",
                        sep=" ",
                        skipinitialspace=True
                )
                dataset = raw_dataset.copy()
                dataset.tail()
                dataset = dataset.dropna()

                # Replace origin column with multiple columns for each location
                dataset["Origin"] = dataset["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
                dataset = pd.get_dummies(dataset, columns=["Origin"], prefix="", prefix_sep="")
                dataset.tail()

                # Split data into training and validation segments
                train_dataset = dataset.sample(frac=0.8, random_state=0)
                valid_dataset = dataset.drop(train_dataset.index)

                # Copy the data to a new set of variables
                self.__train_features = train_dataset.copy()
                self.__valid_features = valid_dataset.copy()

                # Split the output (target variable) into its own label variable
                self.__train_labels = self.__train_features.pop("MPG")
                self.__valid_labels = self.__valid_features.pop("MPG")

        def __compile_model(self, hp):
                # Normalise the features
                normalizer = tf.keras.layers.Normalization(axis=-1)
                normalizer.adapt(np.array(self.__train_features))
                
                # Create non-linear model
                model = tf.keras.Sequential([normalizer])

                for i in range(hp.Int("layers", 0, 5)):
                       model.add(tf.keras.layers.Dense(units=hp.Int("units_" + str(i), 32, 256, step=32), activation="relu"))
                model.add(tf.keras.layers.Dense(1))
               
                model.summary()

                # Tune the learning rate for the optimizer
                # Choose an optimal value from 0.01, 0.001, or 0.0001
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
                        directory="my_dir",
                        project_name="intro_to_kt"
                )
                print(tuner.search_space_summary())

                tuner.search(
                        self.__train_features,
                        self.__train_labels,
                        epochs=50,
                        validation_data=(self.__valid_features, self.__valid_labels),
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)]
                )

                # Get the optimal hyperparameters
                best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
                
                # Build the model with the optimal hyperparameters and train it on the data for 50 epochs
                model = tuner.hypermodel.build(best_hps)
                history = self.__train_model(model)

                val_loss_per_epoch = history.history["val_loss"]               
                best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
                print("Best epoch: %d" % (best_epoch))

                model.save("BestModel")

        def __train_model(self, model):
                history = model.fit(
                        self.__train_features,
                        self.__train_labels,
                        epochs=50,
                        validation_data=(self.__valid_features, self.__valid_labels),
                        verbose=0 # Suppress logging
                )
                self.__plot_loss(history)
                return history

        def __plot_loss(self, history):
                plt.plot(history.history["loss"], label="loss")
                plt.plot(history.history["val_loss"], label="val_loss")
                plt.ylim([0, 10])
                plt.xlabel("Epoch")
                plt.ylabel("Error [MPG]")
                plt.legend()
                plt.grid(True)
                plt.show()