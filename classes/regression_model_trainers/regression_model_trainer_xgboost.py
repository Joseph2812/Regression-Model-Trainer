import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from classes.regression_model_trainers.regression_model_trainer import RegressionModelTrainer as RMTrainer

class RegressionModelTrainerXGBoost(RMTrainer):
    RESULTS_CONTENT = "<XGBoost>: Best_Score={score:f}, Best_NTree_Limit={ntree_limit:d}, ETA={eta:f}."

    OBJECTIVE = "reg:squarederror"
    EVALUATION_METRIC = "rmse"
    
    EARLY_STOPPING_ROUNDS = 5

    # GBTree Parameters
    ETA = 0.3 # Learning rate

    def __init__(self):
        super().__init__()
        self._set_trainer_name("XGBoost")

        xgb.config_context(verbosity=2)
        
    def _train_and_save_best_model(self) -> tuple[str, list[float], list[float], list[float]]:
        print("[XGBoost] Training model...")

        model = self.__train_model()
        results = model.evals_result()

        (best_epoch, lowest_val_loss) = self._get_best_epoch_and_val_loss(results["validation_1"]["rmse"])

        predictions = model.predict(self._all_selected_features)
        test_predictions = model.predict(self._selected_test_features)

        model.save_model(self._model_dir.format(epoch=best_epoch, val_loss=lowest_val_loss) + ".model")
        self._save_results(
            epoch=best_epoch,
            loss=results["validation_0"]["rmse"][best_epoch - 1],
            val_loss=lowest_val_loss,
            test_loss=mean_absolute_error(self._data["test"]["labels"], test_predictions),
            unique_results=self.RESULTS_CONTENT.format(
                score=model.best_score,
                ntree_limit=model.best_ntree_limit,
                eta=self.ETA
            )          
        )

        return (self.EVALUATION_METRIC, results["validation_0"]["rmse"], results["validation_1"]["rmse"], predictions)

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
