import xgboost as xgb
from sklearn.metrics import mean_squared_error
from classes.regression_model_trainers.regression_model_trainer import RegressionModelTrainer as RMTrainer
from classes.live_plotter.plotter_process_manager import PlotterProcessManager

class RegressionModelTrainerXGBoost(RMTrainer):
    RESULTS_CONTENT = "<XGBoost>: BestScore={score:f}, BestNTreeLimit={ntree_limit:d}, ETA={eta:f}."

    OBJECTIVE = "reg:squarederror"
    EVALUATION_METRIC = "rmse"
    
    EARLY_STOPPING_ROUNDS = 5

    # GBTree parameters #
    ETA = 0.3 # Learning rate

    def __init__(self):
        super().__init__()
        self._set_trainer_name("XGBoost")

        xgb.set_config(verbosity=2)
        
    def _train_and_save_best_model(self) -> tuple[list[float], list[float], float, list[float], int, float, str, str]:
        print("[XGBoost] Training model...")

        model = self.__train_model()
        results = model.evals_result()

        predictions = model.predict(self._all_selected_features)
        test_predictions = model.predict(self._selected_test_features)

        (best_epoch, best_val_loss) = self._get_best_epoch_and_val_loss(results["validation_1"][self.EVALUATION_METRIC])
        model.save_model(self._model_dir.format(epoch=best_epoch, val_loss=best_val_loss) + ".model")

        return (
            results["validation_0"][self.EVALUATION_METRIC],
            results["validation_1"][self.EVALUATION_METRIC],
            mean_squared_error(self._data["test"]["labels"], test_predictions, squared=False),
            predictions,
            best_epoch,
            best_val_loss,
            self.EVALUATION_METRIC,
            self.RESULTS_CONTENT.format(
                score=model.best_score,
                ntree_limit=model.best_ntree_limit,
                eta=self.ETA
            )
        )

    def __train_model(self) -> xgb.XGBModel:
        plot_manager = PlotterProcessManager("Validation Loss at Current Hyperparameters", "Epoch", self.EVALUATION_METRIC)

        model = xgb.XGBRegressor(
            objective=self.OBJECTIVE,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            eval_metric=self.EVALUATION_METRIC,
            booster="gbtree", # gbtree, gblinear, or dart
            verbosity=1,
            callbacks=[PlotCallback(plot_manager, self.EVALUATION_METRIC)]
        )
        model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            verbose=True,
            eval_set=[(self._selected_train_features, self._data["train"]["labels"]), (self._selected_valid_features, self._data["valid"]["labels"])],
        )
        # Early stopping will use the last eval_set & eval_metric in the list

        plot_manager.end_process()
        return model

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