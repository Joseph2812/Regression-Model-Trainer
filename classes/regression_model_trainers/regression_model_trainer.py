import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import shutil
import matplotlib.pyplot as plt
import pandas as pd

class RegressionModelTrainer:
    """
    This is the abstract base class, instantiate a derived class to choose a specific machine learning library.\n
    Trains on a given dataset, and tries to find the optimal model by minimising validation loss.\n
    Use start_training() to find the best model with your desired features.

    Data Split = 80:10:10 (Training:Validation:Testing).
    """

    RESULTS_CONTENT_START = "[Training Session {train_session:d}]: Epoch={epoch:d}, Loss={loss:f}, Validation_Loss={val_loss:f}, Test_Loss(MAE)={test_loss:f}. "

    MASTER_ROOT = "Output"
    
    # Current directory: MASTER_ROOT\MODELS_ROOT\__trainer_name\SESSION_NAME\MODEL_FILENAME
    MODELS_ROOT    = os.path.join(MASTER_ROOT, "BestModels")
    SESSION_NAME   = "TrainingSession_{count:d}"
    MODEL_FILENAME = "model_E{epoch:02d}-VL{val_loss:f}"

    PLOTS_ROOT = os.path.join(MASTER_ROOT, "Plots")
    DATA_ROOT = os.path.join(MASTER_ROOT, "Data")
    RESULTS_DIRECTORY = os.path.join(MASTER_ROOT, "Results.txt")
    
    # Dictionary structure to store the split up dataset
    _data:dict[str, dict[str, pd.DataFrame | pd.Series]] = None

    @staticmethod
    def set_dataset(data_path:str, label_name:str):
        """Loads a new dataset at the data path.

        Args:
            data_path (str): Path to the dataset in .csv format, relative to this program's location.\n
            label_name (str): Name of the label to train towards, should match the column name in the dataset.
        """

        dataset = pd.read_csv(data_path, header=0)
        RegressionModelTrainer.set_dataset_preloaded(dataset, label_name)

    @staticmethod
    def set_dataset_preloaded(dataset:pd.DataFrame, label_name:str):
        """Loads in a preloaded dataset.

        Args:
            dataset (pandas.DataFrame): Dataset being used for training, validation, and testing data.\n
            label_name (str): Name of the label to train towards, should match the column name in the dataset.
        """

        if RegressionModelTrainer._data == None: RegressionModelTrainer.__setupFilesAndData()

        # Keep all the labels in one variable (for response plot)
        RegressionModelTrainer._all_labels = dataset[label_name]

        # Split data into training, validation, and test segments
        RegressionModelTrainer._data["train"]["features"] = dataset.sample(frac=0.8, random_state=0)
        remainder = dataset.drop(RegressionModelTrainer._data["train"]["features"].index) # 20% Remainder

        RegressionModelTrainer._data["valid"]["features"] = remainder.sample(frac=0.5, random_state=0)
        RegressionModelTrainer._data["test"]["features"] = remainder.drop(RegressionModelTrainer._data["valid"]["features"].index)

        # Split the labels into their own entry
        RegressionModelTrainer._data["train"]["labels"] = RegressionModelTrainer._data["train"]["features"].pop(label_name)
        RegressionModelTrainer._data["valid"]["labels"] = RegressionModelTrainer._data["valid"]["features"].pop(label_name)
        RegressionModelTrainer._data["test"]["labels"] = RegressionModelTrainer._data["test"]["features"].pop(label_name)

    @staticmethod
    def __setupFilesAndData():
        if os.path.exists(RegressionModelTrainer.MASTER_ROOT):
            shutil.rmtree(RegressionModelTrainer.MASTER_ROOT) # Clear previous stuff
        os.mkdir(RegressionModelTrainer.MASTER_ROOT)

        # ---Setting up directories---
        os.mkdir(RegressionModelTrainer.MODELS_ROOT)
        os.mkdir(RegressionModelTrainer.DATA_ROOT)
        os.mkdir(RegressionModelTrainer.PLOTS_ROOT)           

        with open(RegressionModelTrainer.RESULTS_DIRECTORY, 'w') as f:
            f.write("=====Best Models=====") # Creates txt file

        RegressionModelTrainer._data = {
            "train": {"features": [], "labels": []},
            "valid": {"features": [], "labels": []},
            "test" : {"features": [], "labels": []}
        }

    def __new__(cls, *args, **kwargs):
        if cls is RegressionModelTrainer:
            raise TypeError("Base class may not be instantiated")
        return super().__new__(cls)

    def __init__(self):
        if (self._data == None): raise RuntimeError("A shared dataset must be assigned first. Use RegressionModelTrainer.set_dataset().")

        self._training_count:int = 0 # Tracks how many times start_training() is run, so it can name the folder accordingly
        self._model_dir:str # Where to save models

        # For feature selection
        self._selected_train_features:pd.DataFrame
        self._selected_valid_features:pd.DataFrame
        self._selected_test_features:pd.DataFrame

        self._all_selected_features:pd.DataFrame
        self._all_labels:pd.DataFrame

        # Assigned to by derived classes through _set_trainer_name()
        self.__trainer_name:str
        self.__data_dir:str
        self.__plots_dir:str

    def start_training(self, selected_columns:list[str]=[]):
        """Start trialling various models, and fully train the best model found.
        
        Args:
            selected_columns (str): Columns that you want to include in training, leave empty to use all of the columns. Default = [].
        """

        print("\n=====Training Session {:d}=====".format(self._training_count))
                
        # Update directory path for this training session
        session_dir = os.path.join(self.MODELS_ROOT, self.__trainer_name, self.SESSION_NAME.format(count=self._training_count))
        os.mkdir(session_dir)

        self._model_dir = os.path.join(session_dir, self.MODEL_FILENAME)
        
        # Select requested feature columns
        if selected_columns == []:
            self._selected_train_features = self._data["train"]["features"]
            self._selected_valid_features = self._data["valid"]["features"]
            self._selected_test_features = self._data["test"]["features"]
        else:
            self._selected_train_features = self._data["train"]["features"][selected_columns]
            self._selected_valid_features = self._data["valid"]["features"][selected_columns]
            self._selected_test_features = self._data["test"]["features"][selected_columns]

        self._all_selected_features = pd.concat([self._selected_train_features, self._selected_valid_features, self._selected_test_features])
        self._all_selected_features.sort_index()

        # Show selected data (preview to see if it's setup right)
        print("\n---Selected Training Data---")
        print(self._selected_train_features.head(), end="\n\n")
        print(self._data["train"]["labels"].head())

        print("\n---Selected Validation Data---")
        print(self._selected_valid_features.head(), end="\n\n")
        print(self._data["valid"]["labels"].head(), end="\n\n")

        print("\n---Selected Testing Data---")
        print(self._selected_test_features.head(), end="\n\n")
        print(self._data["test"]["labels"].head(), end="\n\n")

        self.__analyse_best_model()

        self._training_count += 1

    def _set_trainer_name(self, trainer_name:str):
        """A derived class should call this in __init__() to set the trainer's name"""
        self.__trainer_name = trainer_name

        os.mkdir(os.path.join(self.MODELS_ROOT, trainer_name))

        self.__data_dir = os.path.join(self.DATA_ROOT, trainer_name)
        os.mkdir(self.__data_dir)

        # Make trainer's plot directory
        self.__plots_dir = os.path.join(self.PLOTS_ROOT, trainer_name)
        os.mkdir(self.__plots_dir)

    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        """Derived class must override this with it's own method for training & saving models.\n       
        At the end it must return characteristics for printing and plotting (analysis).

        Returns: (eval_metric:str, losses:list[float], val_losses:list[float], predictions:list[float])
        """
        raise NotImplementedError("Method: ""_train_and_save_best_model"" not implemented")

    def _get_best_epoch_and_val_loss(self, val_loss:list[float]) -> tuple[int, float]:
        best_epoch = val_loss.index(min(val_loss)) + 1
        lowest_val_loss = val_loss[best_epoch - 1]

        return (best_epoch, lowest_val_loss)

    def _save_results(self, epoch:int, loss:float, val_loss:float, test_loss:float, unique_results:str):
        with open(self.RESULTS_DIRECTORY, 'a') as f:
            f.write('\n' +
                self.RESULTS_CONTENT_START.format(
                    train_session=self._training_count,
                    epoch=epoch,
                    loss=loss,
                    val_loss=val_loss,
                    test_loss=test_loss
                ) + unique_results
            )

    def __analyse_best_model(self):
        (eval_metric, losses, val_losses, predictions) = self._train_and_save_best_model()

        print("\n=====Best Model=====")

        # Print out which has the lowest validation loss at the best epoch       
        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(val_losses)
        print("\nBest epoch: {:d}".format(best_epoch))
        print("Lowest validation loss: {:f}".format(lowest_val_loss))

        # Save loss & val_loss data
        with open(os.path.join(self.__data_dir, self.SESSION_NAME.format(count=self._training_count) + "_Loss.csv"), 'w') as f:
            f.write("Epoch,loss,val_loss\n")
            for i in range(len(losses)):
                f.write(f"{i+1},{losses[i]},{val_losses[i]}")
                if i != len(losses) - 1: f.write('\n')

        # Save response data
        with open(os.path.join(self.__data_dir, self.SESSION_NAME.format(count=self._training_count) + "_Response.csv"), 'w') as f:
            f.write("Actual,Predicted\n")
            for i in range(len(self._all_labels)):
                f.write(f"{self._all_labels[i]},{float(predictions[i])}") # float() removes square brackets
                if i != len(self._all_labels) - 1: f.write('\n')

        self.__plot_model(eval_metric, losses, val_losses, predictions)

    def __plot_model(self, eval_metric:str, losses:list[float], val_losses:list[float], predictions:list[float]):
        session_name = self.SESSION_NAME.format(count=self._training_count)
        
        general_title = f"[{self.__trainer_name}] " + session_name
        general_dir = os.path.join(self.__plots_dir, session_name + "{0}.png")

        # Plots loss and val_loss
        fig = plt.figure()
        plt.plot(list(range(1, len(losses) + 1)), losses, label="loss") # Epochs start at 1
        plt.plot(list(range(1, len(val_losses) + 1)), val_losses, label="val_loss")
        plt.xlim([0, len(losses) + 1]) # +1 padding
        plt.ylim([0, max(losses) * 1.1]) # 10% padding
        plt.title(general_title + "_Loss")
        plt.xlabel("Epoch")
        plt.ylabel(eval_metric)
        plt.legend()
        plt.grid(True)
        fig.savefig(general_dir.format("_Loss"))

        # Plots response
        fig = plt.figure()
        plt.plot(predictions, label="Predictions")
        plt.plot(self._all_labels, label="Actual")
        plt.xlim([0, len(predictions)])
        plt.title(general_title + "_Response")
        plt.xlabel("Datapoint")
        plt.ylabel(self._data["train"]["labels"].name)
        plt.legend()
        plt.grid(True)
        fig.savefig(general_dir.format("_Response"))