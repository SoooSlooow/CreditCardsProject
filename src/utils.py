from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures


class RatioFeatures:

    def __init__(self) -> None:
        self.n_of_features = None
        self.n_of_input_features = None

    def fit(self, X: np.ndarray, y: None = None) -> None:
        self.n_of_features = 0
        self.n_of_input_features = X.shape[1]
        for i in range(self.n_of_input_features):
            for j in range(i + 1, self.n_of_input_features):
                self.n_of_features += 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X)
        new_features = np.zeros((X.shape[0], self.n_of_features))
        c = 0
        for i in range(self.n_of_input_features):
            for j in range(i + 1, self.n_of_input_features):
                new_features[:, c] = X[:, i] / (X[:, j] + 0.01)
                c += 1
        return np.concatenate((X, new_features), axis=1)


def create_column_transformers(X: pd.DataFrame) -> dict[str, ColumnTransformer]:
    num_vars = [var for var in X.columns.values if X[var].dtype != 'object' and var != 'BAD_CLIENT']
    cat_vars = [var for var in X.columns.values if X[var].dtype == 'object']
    cat_vars_with_ot_dropped = [var for var in cat_vars if var != 'OCCUPATION_TYPE']

    categorical_transformation_1 = Pipeline(
        [('ohe', OneHotEncoder(sparse=False, drop='if_binary')), ('scaler', StandardScaler())])
    numerical_transformation_1 = StandardScaler()
    ct_simple_transformer = ColumnTransformer([('cat_transformation', categorical_transformation_1, cat_vars),
                                               ('num_transformation', numerical_transformation_1, num_vars)])

    categorical_transformation_2 = Pipeline(
        [('ohe', OneHotEncoder(sparse=False, drop='if_binary')), ('scaler', StandardScaler())])
    numerical_transformation_2 = Pipeline([('polyfeat', PolynomialFeatures(degree=2)),
                                           ('scaler', StandardScaler())
                                           ])
    ct_with_poly_features = ColumnTransformer([('cat_transformation', categorical_transformation_2, cat_vars),
                                               ('num_transformation', numerical_transformation_2, num_vars)])

    categorical_transformation_3 = Pipeline(
        [('ohe', OneHotEncoder(sparse=False, drop='if_binary')), ('scaler', StandardScaler())])
    numerical_transformation_3 = Pipeline([('ratfeat', RatioFeatures()),
                                           ('scaler', StandardScaler())
                                           ])
    ct_with_ratio_features = ColumnTransformer([('cat_transformation', categorical_transformation_3, cat_vars),
                                                ('num_transformation', numerical_transformation_3, num_vars)])

    categorical_transformation_4 = Pipeline(
        [('ohe', OneHotEncoder(sparse=False, drop='if_binary')), ('scaler', StandardScaler())])
    numerical_transformation_4 = Pipeline([('ratfeat', RatioFeatures()),
                                           ('polyfeat', PolynomialFeatures(degree=2)),
                                           ('scaler', StandardScaler())
                                           ])
    ct_with_ratio_and_poly_features = ColumnTransformer([('cat_transformation', categorical_transformation_4, cat_vars),
                                                         ('num_transformation', numerical_transformation_4, num_vars)])

    categorical_transformation_5 = Pipeline(
        [('ohe', OneHotEncoder(sparse=False, drop='if_binary')), ('scaler', StandardScaler())])
    numerical_transformation_5 = StandardScaler()

    ct_without_occupation_type = ColumnTransformer(
        [('cat_transformation', categorical_transformation_5, cat_vars_with_ot_dropped),
         ('num_transformation', numerical_transformation_5, num_vars)])

    column_transformers = {'simple_transformer': ct_simple_transformer,
                           'with_poly_features': ct_with_poly_features,
                           'with_ratio_features': ct_with_ratio_features,
                           # 'with_ratio_and_poly_features': ct_with_ratio_and_poly_features,
                           'without_occupation_type': ct_without_occupation_type
                           }

    return column_transformers


def create_feature_selectors() -> dict[int: SelectPercentile]:
    feature_selectors = {percentile: SelectPercentile(score_func=f_classif,
                                                      percentile=percentile) for percentile in range(10, 101, 10)}
    return feature_selectors


def create_pipeline(
        column_transformer: ColumnTransformer,
        classifier: Any,
        feature_selector: Any = 'passthrough'
) -> Pipeline:
    return Pipeline([('column_transformer', column_transformer),
                     ('feature_selector', feature_selector),
                     ('classifier', classifier)])


def get_features_transformation_info(df: pd.DataFrame, classifier: Any, cv: Any) -> pd.DataFrame:
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']

    column_transformers = create_column_transformers(X)
    feature_selectors = create_feature_selectors()

    ft_info = pd.DataFrame(data=np.zeros((len(feature_selectors), len(column_transformers))),
                           index=feature_selectors.keys(),
                           columns=column_transformers.keys())
    for ct in column_transformers:
        for fs in feature_selectors:
            cv_score = cross_val_score(create_pipeline(column_transformers[ct],
                                                       classifier,
                                                       feature_selectors[fs]),
                                       X,
                                       y,
                                       cv=cv,
                                       scoring='roc_auc')
            ft_info.loc[fs, ct] = cv_score.mean()
    return ft_info
