import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class RegressionModelTrainer:
    """
    This is the abstract base class, instantiate a derived class to choose a specific machine learning library.\n
    Trains on a given dataset, and tries to find the optimal model by minimising validation loss.\n
    Use start_tuning() to find the best model with your desired features.
    """

    RESULTS_CONTENT_START = "[Training Session {:d}]: "

    # Current directory: MODELS_ROOT\_trainer_name\SESSION_NAME\MODEL_FILENAME
    MODELS_ROOT         = "BestModels"
    SESSION_NAME        = "TrainingSession_{count:d}"
    MODEL_FILENAME      = "model_E{epoch:02d}-VL{val_loss:f}"

    PLOTS_ROOT          = "ModelPlots"
    PLOTS_DIRECTORY     = PLOTS_ROOT + "\\{trainer}"
    PLOT_MODEL          = True

    RESULTS_FILENAME    = "Results.txt"
    
    def __new__(cls, *args, **kwargs):
        if cls is RegressionModelTrainer:
            raise TypeError("Base class may not be instantiated")
        return super().__new__(cls)

    def __init__(self, data_path:str, label_name:str):
        """Initialises the trainer with a dataset.

        Args:
            data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
            label_name (str): Name of the label to train towards, should match the column name in the dataset.\n
        """

        self._training_count = 0 # Tracks how many times start_tuning() is run, so it can name the folder accordingly
        self._current_dir = ""

        # Dictionary structure to store the split up dataset
        self._data = {
            "train": {"features": [], "labels": []},
            "valid": {"features": [], "labels": []}
        }

        # For feature selection
        self._selected_train_features = []
        self._selected_valid_features = []
        
        # Assigned to by derived classes through _set_trainer_name()
        self.__trainer_name = "" 
        self.__plots_directory = ""        

        # Load initial dataset
        self.change_dataset(data_path, label_name)
    
    @staticmethod
    def reset_files():
        # ---Clearing and setting up directories---
        if os.path.exists(RegressionModelTrainer.MODELS_ROOT):
            shutil.rmtree(RegressionModelTrainer.MODELS_ROOT) # Clear previous models
        os.mkdir(RegressionModelTrainer.MODELS_ROOT)

        if os.path.exists(RegressionModelTrainer.PLOTS_ROOT):
            shutil.rmtree(RegressionModelTrainer.PLOTS_ROOT) # Clear previous plots
        if RegressionModelTrainer.PLOT_MODEL:
            os.mkdir(RegressionModelTrainer.PLOTS_ROOT)

        with open(RegressionModelTrainer.RESULTS_FILENAME, 'w') as f:
            f.write("=====Best Models=====") # Creates txt file or overwrites an existing one

    def change_dataset(self, data_path:str, label_name:str):
        """Loads a new dataset.

        Args:
            data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
            label_name (str): Name of the label to train towards, should match the column name in the dataset.
        """
        
        dataset = pd.read_csv(data_path, header=0)

        # Split data into training and validation segments
        self._data["train"]["features"] = dataset.sample(frac=0.8, random_state=0)
        self._data["valid"]["features"] = dataset.drop(self._data["train"]["features"].index)

        # Split the labels into their own entry
        self._data["train"]["labels"] = self._data["train"]["features"].pop(label_name)
        self._data["valid"]["labels"] = self._data["valid"]["features"].pop(label_name)

    def start_training(self, selected_columns:list[str]=[]):
        """Start trialling various models, and fully train the best model found.
        
        Args:
            selected_columns (str): Columns that you want to include in training, leave empty to use all of the columns. Default = [].
        """

        print("\n=====Training Session {:d}=====".format(self._training_count))
                
        # Update directory path for this training session
        session_dir = self.MODELS_ROOT + '\\' + self.__trainer_name + '\\' + self.SESSION_NAME.format(count=self._training_count)
        self._current_dir = os.getcwd() + '\\' + session_dir + '\\' + self.MODEL_FILENAME
        
        # Create new session folder
        os.mkdir(session_dir)

        # Select requested feature columns
        if selected_columns == []:
            self._selected_train_features = self._data["train"]["features"]
            self._selected_valid_features = self._data["valid"]["features"]
        else:
            self._selected_train_features = self._data["train"]["features"][selected_columns]
            self._selected_valid_features = self._data["valid"]["features"][selected_columns]

        # Show selected data (preview to see if it's setup right)
        print("\n---Selected Training Data---")
        print(self._selected_train_features.head(), end="\n\n")
        print(self._data["train"]["labels"].head())

        print("\n---Selected Validation Data---")
        print(self._selected_valid_features.head(), end="\n\n")
        print(self._data["valid"]["labels"].head(), end="\n\n")

        self.__analyse_best_model()

        self._training_count += 1

    def _set_trainer_name(self, trainer_name:str):
        """A derived class should call this in __init__() to set the trainer's name"""
        self.__trainer_name = trainer_name
        os.mkdir(self.MODELS_ROOT + '\\' + self.__trainer_name)

        # Make trainer's plot directory
        self.__plots_directory = self.PLOTS_ROOT + '\\' + trainer_name
        if self.PLOT_MODEL:
            os.mkdir(self.__plots_directory)

    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        """Derived class with override this with it's own method for training & saving models.\n       
        At the end it must return characteristics for printing and plotting (analysis)

        Returns: (losses:list[float], val_losses:list[float], predictions:list[float] of 5)
        """
        raise NotImplementedError("Method: ""_train_and_save_best_model"" not implemented")

    def _get_best_epoch_and_val_loss(self, val_loss:list[float]) -> tuple[int, float]:
        best_epoch = val_loss.index(min(val_loss)) + 1
        lowest_val_loss = val_loss[best_epoch - 1]

        return (best_epoch, lowest_val_loss)

    def _save_results(self, results:str):
        with open(self.RESULTS_FILENAME, 'a') as f:
            f.write('\n' + self.RESULTS_CONTENT_START.format(self._training_count) + results)

    def __analyse_best_model(self):
        (eval_metric, losses, val_losses, predictions) = self._train_and_save_best_model()

        print("\n=====Best Model=====")

        # Print out which has the lowest validation loss at the best epoch       
        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(val_losses)
        print("\nBest epoch: {:d}".format(best_epoch))
        print("Lowest validation loss: {:f}".format(lowest_val_loss))
        
        self.__print_prediction_preview(predictions)

        if self.PLOT_MODEL:
            self.__plot_model(eval_metric, losses, val_losses)

    def __print_prediction_preview(self, predictions:list[float]):
        print("\nPredicting with 5 validation features to preview the label:\n\n---Validation Features---")
        print(self._selected_valid_features.head())

        label_name = self._data["valid"]["labels"].name

        print("\n---Predicted Labels ({})---".format(label_name))
        print(predictions)

        print("\n---Validation Labels ({})---".format(label_name))
        print(self._data["valid"]["labels"].head(), end="\n\n")

    def __plot_model(self, eval_metric:str, losses:list[float], val_losses:list[float]):
        session_name = self.SESSION_NAME.format(count=self._training_count)

        # Plots loss and val_loss
        fig = plt.figure()
        plt.plot(list(range(1, len(losses) + 1)), losses, label="loss")
        plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label="val_loss")
        plt.xlim([0, len(losses) + 1]) # +1 padding
        plt.ylim([0, max(losses) * 1.1]) # 10% padding
        plt.title(f"[{self.__trainer_name}] " + session_name)
        plt.xlabel("Epoch")
        plt.ylabel(eval_metric)
        plt.legend()
        plt.grid(True)
        fig.savefig(self.__plots_directory + '\\' + f"{session_name}.png")
