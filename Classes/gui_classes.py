import json
import os
import pandas as pd
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from Classes.regression_model_trainer import RegressionModelTrainer as RMTrainer
from Classes.regression_model_trainer_tensorflow import RegressionModelTrainerTensorFlow as RMTrainerTF
from Classes.regression_model_trainer_xgboost import RegressionModelTrainerXGBoost as RMTrainerXGB

class RootWindow(Tk):
    CONFIG_DIRECTORY = "config.json"

    dataset:pd.DataFrame
    config = {
        "algorithms": {
            "using_tensorflow": True,
            "using_xgboost": False
        },
        "data_path": "",
        "label": "",
        "feature_lists": [], # list[list[str]]
        "parameters": {
            "tensorflow": RMTrainerTF.parameters
        }
    }
    train_button:ttk.Button

    def __init__(self):
        super().__init__()

        # Load saved config
        if os.path.exists(self.CONFIG_DIRECTORY):
            with open(self.CONFIG_DIRECTORY, 'r') as f:
                RootWindow.config = json.load(f)

        # Cache original values to check for changes when beginning training
        self.__old_feature_lists:list[list[str]] = self.config["feature_lists"].copy()
        self.__old_tf_parameters:dict[str, any] = self.config["parameters"]["tensorflow"].copy()

        self.title("Regression Model Trainer Setup")
        self.resizable(width=False, height=False)

        ttk.Button(self, text="Configure Trainers", command=lambda:ConfigureWindow(self)).grid(column=0, row=0, pady=10)
        RootWindow.train_button = ttk.Button(self, text="Train", command=self.__load_into_training, state="disabled")
        self.train_button.grid(column=0, row=5)

        AlgorithmFrame(self)
        DataPathFrame(self)

        self.mainloop()

    def destroy(self):
        RootWindow.train_button = None
        self.__save_config()

        super().destroy()
        quit()

    def __load_into_training(self):
        self.__save_config()
        self.train_button.configure(state="disabled")

        try:
            rmt:RMTrainer
            RMTrainer.set_dataset_preloaded(self.dataset, self.config["label"])

            if self.config["algorithms"]["using_tensorflow"]:
                RMTrainerTF.parameters = self.config["parameters"]["tensorflow"]
                
                # If features or parameters change, then clear previous trials.                            
                rmt = RMTrainerTF(self.__old_feature_lists == self.config["feature_lists"] and self.__old_tf_parameters == self.config["parameters"]["tensorflow"])

                for feature_list in self.config["feature_lists"]:
                    rmt.start_training(feature_list)

            if self.config["algorithms"]["using_xgboost"]:
                #RMTrainerXGB.parameters = self.config["parameters"]["xgboost"]
                rmt = RMTrainerXGB()
                for feature_list in self.config["feature_lists"]:
                    rmt.start_training(feature_list)
        finally:
            self.destroy()

    def __save_config(self):
        with open(self.CONFIG_DIRECTORY, 'w') as f:
            json.dump(self.config, f, indent=4)

class AlgorithmFrame(ttk.Labelframe):
    def __init__(self, master):
        super().__init__(master, text="Algorithms to use")
        
        self.__using_tensorflow_var = BooleanVar(self, RootWindow.config["algorithms"]["using_tensorflow"])
        self.__using_xgboost_var = BooleanVar(self, RootWindow.config["algorithms"]["using_xgboost"])

        self.grid(column=0, row=1)

        ttk.Checkbutton(self, text="TensorFlow", variable=self.__using_tensorflow_var, offvalue=False, onvalue=True).pack(side=LEFT)
        ttk.Checkbutton(self, text="XGBoost", variable=self.__using_xgboost_var, offvalue=False, onvalue=True).pack(side=LEFT)

        self.__using_tensorflow_var.trace_add("write", self.__update_tf_config)
        self.__using_xgboost_var.trace_add("write", self.__update_xgb_config)

    def __update_tf_config(self, *args): RootWindow.config["algorithms"]["using_tensorflow"] = self.__using_tensorflow_var.get()
    def __update_xgb_config(self, *args): RootWindow.config["algorithms"]["using_xgboost"] = self.__using_xgboost_var.get()

class DataPathFrame(ttk.Frame):
    def __init__(self, master:Tk):
        super().__init__(master)

        self.data_path_var = StringVar(self, value=RootWindow.config["data_path"])
        self.data_selection_frame:DataSelectionFrame = None
        self.error_label:ttk.Label = None  

        ttk.Style().configure("Error.TLabel", foreground="red")
        self.error_label = ttk.Label(self, text="Invalid file path (must point to an existing .csv file)", style="Error.TLabel")
        self.error_label.grid(column=1, row=1, padx=10, pady=(0, 10))
        self.error_label.grid_remove()

        self.grid(column=0, row=2)

        vcmd = (self.register(self.__validate_path), '%P')
        ttk.Label(self, text="Data Path:").grid(column=0, row=0, padx=10, pady=10)
        ttk.Entry(self, validate="focusout", validatecommand=vcmd, textvariable=self.data_path_var, width=100).grid(column=1, row=0, padx=(5, 10), pady=10)

        button = ttk.Button(self, text="Browse", command=self.__open_file_explorer)
        button.grid(column=2, row=0, padx=(0, 10), pady=10)

        master.bind("<Return>", lambda _:button.focus_set())

        if RootWindow.config["data_path"] != "":
            self.__validate_path(RootWindow.config["data_path"])

    def __open_file_explorer(self):
        data_path = fd.askopenfilename(
            parent=self.master,
            title="Choose the training data",
            initialdir="./",
            filetypes=[("Comma Separated Values", "*.csv")]
        )
        self.data_path_var.set(data_path)
        self.__validate_path(data_path)
    
    def __validate_path(self, data_path) -> bool:
        try: 
            RootWindow.dataset = pd.read_csv(data_path)
            RootWindow.config["data_path"] = data_path
        except:
            if self.data_selection_frame != None:
                self.data_selection_frame.destroy()
                self.data_selection_frame = None

            self.error_label.grid()
            return False

        self.error_label.grid_remove()

        if self.data_selection_frame != None: self.data_selection_frame.destroy()
        self.data_selection_frame = DataSelectionFrame(self.master)

        return True

class DataSelectionFrame(ttk.Labelframe):
    def __init__(self, master:Tk):
        super().__init__(master, text="Select which label you want the model to predict:", labelanchor='n')   

        self.__model_frame:ModelFrame = None
        self.__column_names:list[str] = RootWindow.dataset.columns.values.tolist()
        self.__selected_label_var = StringVar(self, RootWindow.config["label"])

        self.grid(column=0, row=3, padx=10, pady=(0, 20))

        for name in self.__column_names:
            ttk.Radiobutton(self, text=name, value=name, variable=self.__selected_label_var, command=self.create_model_frame).pack(side=LEFT, padx=10)

        self.__selected_label_var.trace_add("write", self.__update_label_config)

        if self.__selected_label_var.get() != "": self.create_model_frame()       
    
    def destroy(self):
        if self.__model_frame != None: self.__model_frame.destroy()

        super().destroy()
        

    def create_model_frame(self):
        if self.__model_frame != None: self.__model_frame.destroy()

        names_copy = self.__column_names.copy()
        names_copy.remove(self.__selected_label_var.get())
        self.__model_frame = ModelFrame(self.master, names_copy)

    def __update_label_config(self, *args): RootWindow.config["label"] = self.__selected_label_var.get()
    
class ModelFrame(ttk.Labelframe):
    feature_names:list[str]

    class FeatureFrame(ttk.Labelframe):
        def __init__(self, master:ttk.Frame, default_features:list[str]=None, **kwargs):
            super().__init__(master, **kwargs)

            self.check_buttons:list[dict[str, BooleanVar]] = [] # [{name:str, is_active:boolVar}]
            self.__model_frame:ModelFrame = master

            for name in ModelFrame.feature_names:
                if default_features == None:
                    is_active = BooleanVar(self, False)
                else:
                    exists:bool = False
                    for feature in default_features:
                        if name == feature:
                            exists = True
                            break
                    
                    if exists: is_active = BooleanVar(self, True)
                    else:      is_active = BooleanVar(self, False)
                    
                ttk.Checkbutton(self, text=name, offvalue=False, onvalue=True, variable=is_active).pack(side=LEFT, padx=10)

                self.check_buttons.append({"name": name, "is_active_var": is_active})
                is_active.trace_add("write", self.__model_frame.update_features_config)

            ttk.Button(self, text="Select All/None", command=self.__selectOrDeselectBoxes).pack(side=TOP)
            ttk.Button(self, text="Remove", command=self.__remove).pack(side=BOTTOM)

        def __selectOrDeselectBoxes(self):
            hasAllActiveButtons = True
            for button in self.check_buttons:
                if not button["is_active_var"].get():
                    hasAllActiveButtons = False
                    break
            
            if hasAllActiveButtons:
                for button in self.check_buttons: button["is_active_var"].set(False)
            else:
                for button in self.check_buttons: button["is_active_var"].set(True)

        def __remove(self): self.__model_frame.remove_feature_frame(self)

    def __init__(self, master:Tk, feature_names):
        super().__init__(master, text="Select which features you want the model to train with (Select at least two features):", labelanchor='n')

        ModelFrame.feature_names:list[str] = feature_names
        
        self.__feature_frames:list[self.FeatureFrame] = []
        self.__addButton = ttk.Button(self, text="Add Model", command=self.__add_new_feature_frame)
        
        self.grid(column=0, row=4, padx=10, pady=20)
        self.__addButton.pack(side=BOTTOM, pady=(0, 10))

        RootWindow.train_button.configure(state="normal")

        for feature_list in RootWindow.config["feature_lists"]:
            self.__add_new_feature_frame(feature_list)
              
    def update_features_config(self, *args):
        feature_lists:list[list[str]] = []

        for frame in self.__feature_frames:
            feature_list:list[str] = []

            for button in frame.check_buttons:
                if button["is_active_var"].get():
                    feature_list.append(button["name"])
            
            feature_lists.append(feature_list)
        
        RootWindow.config["feature_lists"] = feature_lists

    def remove_feature_frame(self, feature_frame):
        self.__feature_frames.remove(feature_frame)
        feature_frame.destroy()

        for i, frame in enumerate(self.__feature_frames):
            frame.configure(text=f"TrainingSession_{i} Features")

        self.update_features_config()

    def __add_new_feature_frame(self, default_features:list[str]=None):
        feature_frame = self.FeatureFrame(self, default_features, text=f"TrainingSession_{len(self.__feature_frames)} Features")
        self.__feature_frames.append(feature_frame)

        feature_frame.pack(side=TOP, pady=10)

        self.update_features_config()

    def destroy(self):
        if RootWindow.train_button != None:
            RootWindow.train_button.configure(state="disabled")   

        super().destroy()

# ---For configuration window---
class ConfigureWindow(Toplevel):
    def __init__(self, master:Tk):
        super().__init__(master)

        self.title("Trainer Configurations")
        self.grab_set()

        notebook = ttk.Notebook(self)
        tf_tab = TensorFlowFrame(notebook)
        xgb_tab = ttk.Frame(notebook)
        notebook.add(tf_tab, text="TensorFlow")
        notebook.add(xgb_tab, text="XGBoost")
        notebook.pack()

        ttk.Label(xgb_tab, text="Custom configuration has not been implemented yet.\nCan still be used at the default settings.").pack()

class TensorFlowFrame(ttk.Frame):
    class ObjectiveFrame(ttk.Frame):
        def __init__(self, master:ttk.Frame, parameters):
            super().__init__(master)

            self.objective_var = StringVar(self, parameters["objective"])

            ttk.Label(self, text="Objective:")              .grid(column=0, row=0, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.objective_var).grid(column=1, row=0, padx=10, pady=5)

    class HyperbandFrame(ttk.Labelframe):
        def __init__(self, master:ttk.Frame, parameters):
            super().__init__(master, text="Hyperband Parameters")

            self.max_epochs_var     = IntVar(self, parameters["max_epochs"])
            self.factor_var         = IntVar(self, parameters["factor"])
            self.hyperband_iter_var = IntVar(self, parameters["hyperband_iterations"])
            self.patience_var       = IntVar(self, parameters["patience"])

            ttk.Label(self, text="Max Epochs:")                  .grid(column=0, row=0, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.max_epochs_var)    .grid(column=1, row=0, padx=10, pady=5)
            
            ttk.Label(self, text="Factor:")                      .grid(column=2, row=0, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.factor_var)        .grid(column=3, row=0, padx=10, pady=5)        
            
            ttk.Label(self, text="Hyperband Iterations:")        .grid(column=0, row=1, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.hyperband_iter_var).grid(column=1, row=1, padx=10, pady=5)

            ttk.Label(self, text="Patience:")                    .grid(column=2, row=1, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.patience_var)      .grid(column=3, row=1, padx=10, pady=5)
    
    class HyperparameterFrame(ttk.LabelFrame):
        def __init__(self, master:ttk.Frame, parameters):
            super().__init__(master, text="Hyperparameter Tuning")

            self.min_hdn_layers_var = IntVar(self, parameters["min_hidden_layers"])
            self.max_hdn_layers_var = IntVar(self, parameters["max_hidden_layers"])

            self.min_units_var = IntVar(self, parameters["min_units"])
            self.max_units_var = IntVar(self, parameters["max_units"])
            self.unit_step_var = IntVar(self, parameters["unit_step"])

            # Change to be length variable later
            self.lrning_rate_choice0_var = DoubleVar(self, parameters["learning_rate_choice"][0])
            self.lrning_rate_choice1_var = DoubleVar(self, parameters["learning_rate_choice"][1])
            self.lrning_rate_choice2_var = DoubleVar(self, parameters["learning_rate_choice"][2])

            ttk.Label(self, text="Minimum Hidden Layers:")            .grid(column=0, row=0, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.min_hdn_layers_var)     .grid(column=1, row=0, padx=10, pady=5)

            ttk.Label(self, text="Maximum Hidden Layers:")            .grid(column=2, row=0, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.max_hdn_layers_var)     .grid(column=3, row=0, padx=10, pady=5)

            ttk.Label(self, text="Minimum Units/Neurons:")            .grid(column=0, row=1, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.min_units_var)          .grid(column=1, row=1, padx=10, pady=5)

            ttk.Label(self, text="Maximum Units/Neurons:")            .grid(column=2, row=1, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.max_units_var)          .grid(column=3, row=1, padx=10, pady=5)

            ttk.Label(self, text="Unit Step:")                        .grid(column=4, row=1, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.unit_step_var)          .grid(column=5, row=1, padx=10, pady=5)

            ttk.Label(self, text="Learning Rate Choice:")             .grid(column=0, row=2, padx=(10, 0), pady=5, sticky=E)
            ttk.Entry(self, textvariable=self.lrning_rate_choice0_var).grid(column=1, row=2, padx=10, pady=5)
            ttk.Entry(self, textvariable=self.lrning_rate_choice1_var).grid(column=2, row=2, padx=10, pady=5)
            ttk.Entry(self, textvariable=self.lrning_rate_choice2_var).grid(column=3, row=2, padx=10, pady=5)

    def __init__(self, master:ttk.Notebook):
        super().__init__(master)

        self.__parameters = RootWindow.config["parameters"]["tensorflow"]

        # Set initial values
        self.__objective_frame = self.ObjectiveFrame(self, self.__parameters)
        self.__hyperband_frame = self.HyperbandFrame(self, self.__parameters)
        self.__hyperparameter_frame = self.HyperparameterFrame(self, self.__parameters)

        # Create & link GUI       
        self.__objective_frame.pack(side=TOP)
        self.__hyperband_frame.pack(side=TOP)
        self.__hyperparameter_frame.pack(side=TOP)

    def destroy(self):
        self.__parameters["objective"] = self.__objective_frame.objective_var.get()

        self.__parameters["max_epochs"]           = self.__hyperband_frame.max_epochs_var.get()
        self.__parameters["factor"]               = self.__hyperband_frame.factor_var.get()
        self.__parameters["hyperband_iterations"] = self.__hyperband_frame.hyperband_iter_var.get()
        self.__parameters["patience"]             = self.__hyperband_frame.patience_var.get()

        self.__parameters["min_hidden_layers"] = self.__hyperparameter_frame.min_hdn_layers_var.get()
        self.__parameters["max_hidden_layers"] = self.__hyperparameter_frame.max_hdn_layers_var.get()

        self.__parameters["unit_step"] = self.__hyperparameter_frame.unit_step_var.get()
        self.__parameters["min_units"] = self.__hyperparameter_frame.min_units_var.get()
        self.__parameters["max_units"] = self.__hyperparameter_frame.max_units_var.get()

        # Change to be length variable later
        self.__parameters["learning_rate_choice"][0] = self.__hyperparameter_frame.lrning_rate_choice0_var.get()
        self.__parameters["learning_rate_choice"][1] = self.__hyperparameter_frame.lrning_rate_choice1_var.get()
        self.__parameters["learning_rate_choice"][2] = self.__hyperparameter_frame.lrning_rate_choice2_var.get()

        super().destroy()