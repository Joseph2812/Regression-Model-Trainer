from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from RegressionModelClasses.regression_model_trainer import RegressionModelTrainer as RMTrainer

class RegressionModelTrainerXGBoost(RMTrainer):
    RESULTS_CONTENT = "<XGBoost>: Epoch={epoch:d}, Loss={loss:f}, Validation_Loss={val_loss:f}, Best_Score={score:f}, Best_NTree_Limit={ntree_limit:f}, ETA={eta:f}"

    OBJECTIVE = "reg:squarederror"
    EVALUATION_METRIC = "rmse"
    
    EARLY_STOPPING_ROUNDS = 5

    # GBTree Parameters
    ETA = 0.3 # Learning rate

    def __init__(self, data_path:str, label_name:str):
        super().__init__(data_path, label_name)
        self._set_trainer_name("XGBoost")

        xgb.config_context(verbosity=2)
        
    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        print("[XGBoost] Training model...")

        model = self.__train_model()
        results = model.evals_result()

        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(results["validation_1"]["rmse"])

        val_predictions = model.predict(self._selected_valid_features)[:5]
        model.save_model(self._current_dir.format(epoch=best_epoch, val_loss=lowest_val_loss) + ".model")

        self._save_results(self.RESULTS_CONTENT.format(
            epoch=model.best_iteration + 1,
            loss=results["validation_1"]["rmse"][best_epoch - 1],
            val_loss=lowest_val_loss,
            score=model.best_score,
            ntree_limit=model.best_ntree_limit,
            eta=self.ETA
        ))

        return (self.EVALUATION_METRIC, results["validation_0"]["rmse"], results["validation_1"]["rmse"], val_predictions)

    def __train_model(self) -> xgb.XGBModel:
        model = xgb.XGBRegressor().set_params(
            objective=self.OBJECTIVE,
            booster="gbtree", # gbtree, gblinear or dart
            verbosity=1
        )
        model.fit(
            self._selected_train_features,
            self._data["train"]["labels"],
            verbose=True,
            early_stopping_rounds=self.EARLY_STOPPING_ROUNDS,
            eval_set=[(self._selected_train_features, self._data["train"]["labels"]), (self._selected_valid_features, self._data["valid"]["labels"])],
            eval_metric=self.EVALUATION_METRIC
        )
        # Early stopping will use the last eval_set & eval_metric in the list

        return model
