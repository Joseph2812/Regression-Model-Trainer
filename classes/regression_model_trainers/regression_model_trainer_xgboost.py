import xgboost as xgb
from sklearn.metrics import mean_squared_error
from classes.regression_model_trainers.regression_model_trainer import RegressionModelTrainer as RMTrainer
from classes.live_plotter.plotter_process_manager import PlotterProcessManager
import hyperopt
from hyperopt import hp
import os
import pickle
import shutil

class RegressionModelTrainerXGBoost(RMTrainer):
    RESULTS_CONTENT = "<XGBoost>: ETA={eta:f}, MaxDepth={max_depth:d}, Gamma={gamma:f}, RegAlpha={reg_alpha:d}, RegLambda={reg_lambda:f}, ColsampleBytree={colsample_bytree:f}, MinChildWeight={min_child_weight:d}."
    TRIALS_DIRECTORY = "xgboost_hyperparameter_trials"

    parameters:dict[str, any] = {
        "objective"  : "reg:squarederror",
        "eval_metric": "rmse",
        "device"     : "gpu:0",

        "max_evals"            : 100,
        "booster"              : "gbtree", # gbtree, gblinear, or dart
        "early_stopping_rounds": 5,
        "seed"                 : 0
    }

    def __init__(self, keep_previous_trials:bool=True):
        """Initialises the XGBoost trainer, and choose whether to resume from previous trials.

        Args:
            keep_previous_trials (bool): Whether you would like the tuner to reuse old trials, for resuming search on the same dataset & tuning parameters. Default = True.
        """

        super().__init__()

        self.__plot_manager:PlotterProcessManager

        self._set_trainer_name("XGBoost")
        xgb.set_config(verbosity=2)

        if not keep_previous_trials:
            if os.path.exists(self.TRIALS_DIRECTORY):
                shutil.rmtree(self.TRIALS_DIRECTORY) # Clear previous trials
            os.mkdir(self.TRIALS_DIRECTORY)
        
    def _train_and_save_best_model(self) -> tuple[list[float], list[float], float, list[float], int, float, str, str]:
        print("[XGBoost] Finding the best hyperparameters...")
        self.__plot_manager = PlotterProcessManager("Validation Loss at Current Hyperparameters", "Epoch", self.parameters["eval_metric"])

        # Tune hyperparameters #
        space = {
            "eta"             : hp.uniform("eta", 0.0, 1.0),
            "max_depth"       : hp.uniformint("max_depth", 3, 18),
            "gamma"           : hp.uniform("gamma", 1.0, 9.0),
            "reg_alpha"       : hp.uniformint("reg_alpha", 40, 180),
            "reg_lambda"      : hp.uniform("reg_lambda", 0.0, 1.0),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            "min_child_weight": hp.uniformint("min_child_weight", 0, 10),
        }

        trials_path = os.path.join(self.TRIALS_DIRECTORY, self._session_name + ".pkl")
        if os.path.exists(trials_path):
            with open(trials_path, "rb") as f:
                trials = pickle.load(f)
        else:
            trials = hyperopt.Trials()

        best_hps = hyperopt.fmin(
            fn               = self.__objective,
            space            = space,
            algo             = hyperopt.tpe.suggest,
            max_evals        = self.parameters["max_evals"],
            trials           = trials,
            trials_save_file = trials_path
        )
        #

        print("[XGBoost] Training the model with the best hyperparameters...")
        model = self.__train_model(best_hps)
        self.__plot_manager.end_process()

        best_epoch = model.best_iteration + 1
        model.save_model(self._model_dir.format(epoch=best_epoch, val_loss=model.best_score) + ".pkl")

        results = model.evals_result()
        predictions = model.predict(self._all_selected_features)
        test_predictions = model.predict(self._selected_test_features)

        return (
            results["validation_0"][self.parameters["eval_metric"]],
            results["validation_1"][self.parameters["eval_metric"]],
            mean_squared_error(self._data["test"]["labels"], test_predictions, squared=False),
            predictions,
            best_epoch,
            model.best_score,
            self.parameters["eval_metric"],
            self.RESULTS_CONTENT.format(
                eta              = best_hps["eta"],
                max_depth        = int(best_hps["max_depth"]),
                gamma            = best_hps["gamma"],
                reg_alpha        = int(best_hps["reg_alpha"]),
                reg_lambda       = best_hps["reg_lambda"],
                colsample_bytree = best_hps["colsample_bytree"],
                min_child_weight = int(best_hps["min_child_weight"])
            )
        )

    def __train_model(self, hps:dict[str, any]) -> xgb.XGBRegressor:
        model = xgb.XGBRegressor(
            objective             = self.parameters["objective"],
            eval_metric           = self.parameters["eval_metric"],
            #device                = self.parameters["device"], # Not available until XGBoost 2.0
            booster               = self.parameters["booster"],
            early_stopping_rounds = self.parameters["early_stopping_rounds"],
            seed                  = self.parameters["seed"],
            verbosity             = 1,
            callbacks             = [PlotCallback(self.__plot_manager, self.parameters["eval_metric"])],

            eta              = hps["eta"],
            max_depth        = int(hps["max_depth"]),
            gamma            = hps["gamma"],
            reg_alpha        = int(hps["reg_alpha"]),
            reg_lambda       = hps["reg_lambda"],
            colsample_bytree = hps["colsample_bytree"],
            min_child_weight = int(hps["min_child_weight"])
        )
        model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            verbose=True,
            eval_set=[(self._selected_train_features, self._data["train"]["labels"]), (self._selected_valid_features, self._data["valid"]["labels"])],
        )
        # Early stopping will use the last eval_set & eval_metric in the list

        return model

    def __objective(self, space:dict[str, any]) -> dict[str, any]:
        model = self.__train_model(space)

        # Return validation loss to minimise
        return {"loss": model.best_score, "status": hyperopt.STATUS_OK}

class PlotCallback(xgb.callback.TrainingCallback):
    def __init__(self, plot_manager:PlotterProcessManager, eval_metric:str):
        super().__init__()

        self.__plot_manager = plot_manager
        self.__eval_metric = eval_metric
    
    def before_training(self, model) -> any:
        self.__plot_manager.plot(None, None)
        return model

    def after_iteration(self, model, epoch, evals_log) -> bool:
        val_losses = evals_log["validation_1"][self.__eval_metric]
        self.__plot_manager.plot(epoch, val_losses[len(val_losses) - 1])
        
        return False