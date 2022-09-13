from typing import Any
import pandas as pd
from src.utils import create_column_transformers, create_pipeline
import os
import optuna
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def optimize_logreg(output_folder: str, df: pd.DataFrame, n_trials: int = 100,
                    cv: Any = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                    ) -> None:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    column_transformers = create_column_transformers(X)

    def objective(trial):
        column_transformer = trial.suggest_categorical('column_transformer', column_transformers.keys())
        logreg_C = trial.suggest_float('logreg_C', 1e-5, 1e4, log=True)
        logreg_penalty = trial.suggest_categorical('logreg_penalty', ['l2'])
        classifier_obj = LogisticRegression(C=logreg_C, penalty=logreg_penalty, solver='lbfgs', random_state=0)
        model_obj = create_pipeline(column_transformers[column_transformer], classifier_obj)
        score = cross_val_score(model_obj, X, y, scoring='roc_auc', cv=cv).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, os.path.join(output_folder, 'study_logreg.pkl'))


def optimize_rf(output_folder: str, df: pd.DataFrame, n_trials: int = 100,
                cv: Any = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                ) -> None:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    column_transformers = create_column_transformers(X)

    def objective(trial):
        column_transformer = trial.suggest_categorical('column_transformer', column_transformers.keys())
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 50, log=True)
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 50, 500)
        classifier_obj = RandomForestClassifier(max_depth=rf_max_depth,
                                                n_estimators=rf_n_estimators,
                                                random_state=0
                                                )
        model_obj = create_pipeline(column_transformers[column_transformer], classifier_obj)
        score = cross_val_score(model_obj, X, y, scoring='roc_auc', cv=cv).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, os.path.join(output_folder, 'study_rf.pkl'))


def optimize_lgbm(output_folder: str, df: pd.DataFrame, n_trials: int = 100,
                  cv: Any = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                  ) -> None:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    column_transformers = create_column_transformers(X)

    def objective(trial):
        column_transformer = trial.suggest_categorical('column_transformer', column_transformers.keys())
        boost_lambda_l2 = trial.suggest_float('boost_lambda_l2', 1e-5, 1e5, log=True)
        boost_max_depth = trial.suggest_int('boost_max_depth', 2, 50)
        boost_n_estimators = trial.suggest_int('boost_n_estimators', 50, 500)
        boost_learning_rate = trial.suggest_float('boost_learning_rate', 0.01, 0.15)
        classifier_obj = LGBMClassifier(lambda_l2=boost_lambda_l2,
                                        max_depth=boost_max_depth,
                                        n_estimators=boost_n_estimators,
                                        learning_rate=boost_learning_rate,
                                        random_state=0
                                        )
        model_obj = create_pipeline(column_transformers[column_transformer], classifier_obj)
        score = cross_val_score(model_obj, X, y, scoring='roc_auc', cv=cv).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, os.path.join(output_folder, 'study_lgbm.pkl'))


def optimize_xgboost(output_folder: str, df: pd.DataFrame, n_trials: int = 100,
                     cv: Any = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                     ) -> None:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    column_transformers = create_column_transformers(X)

    def objective(trial):
        column_transformer = trial.suggest_categorical('column_transformer', column_transformers.keys())
        boost_lambda = trial.suggest_float('boost_lambda', 1e-5, 1e5, log=True)
        boost_max_depth = trial.suggest_int('boost_max_depth', 5, 50)
        boost_n_estimators = trial.suggest_int('boost_n_estimators', 50, 500)
        boost_learning_rate = trial.suggest_float('boost_learning_rate', 0.01, 0.15)
        classifier_obj = XGBClassifier(reg_lambda=boost_lambda,
                                       max_depth=boost_max_depth,
                                       n_estimators=boost_n_estimators,
                                       learning_rate=boost_learning_rate,
                                       random_state=0
                                       )
        model_obj = create_pipeline(column_transformers[column_transformer], classifier_obj)
        score = cross_val_score(model_obj, X, y, scoring='roc_auc', cv=cv).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, os.path.join(output_folder, 'study_xgboost.pkl'))


def optimize_catboost(output_folder: str, df: pd.DataFrame, n_trials: int = 100,
                      cv: Any = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
                      ) -> None:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    column_transformers = create_column_transformers(X)

    def objective(trial):
        column_transformer = trial.suggest_categorical('column_transformer', column_transformers.keys())
        boost_l2_leaf_reg = trial.suggest_float('boost_lambda_l2', 1e-5, 1e5, log=True)
        boost_max_depth = trial.suggest_int('boost_max_depth', 2, 10)
        boost_n_estimators = trial.suggest_int('boost_n_estimators', 50, 500)
        boost_learning_rate = trial.suggest_float('boost_learning_rate', 0.01, 0.15)
        classifier_obj = CatBoostClassifier(l2_leaf_reg=boost_l2_leaf_reg,
                                            max_depth=boost_max_depth,
                                            n_estimators=boost_n_estimators,
                                            learning_rate=boost_learning_rate,
                                            random_state=0
                                            )
        model_obj = create_pipeline(column_transformers[column_transformer], classifier_obj)
        score = cross_val_score(model_obj, X, y, scoring='roc_auc', cv=cv).mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    joblib.dump(study, os.path.join(output_folder, 'study_catboost.pkl'))
