import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


class RatioFeatures:

    def __init__(self):
        self.n_of_features = None
        self.n_of_input_features = None

    def fit(self, X, y=None):
        self.n_of_features = 0
        self.n_of_input_features = X.shape[1]
        for i in range(self.n_of_input_features):
            for j in range(i + 1, self.n_of_input_features):
                self.n_of_features += 1
        return self

    def transform(self, X):
        X = np.array(X)
        new_features = np.zeros((X.shape[0], self.n_of_features))
        c = 0
        for i in range(self.n_of_input_features):
            for j in range(i + 1, self.n_of_input_features):
                new_features[:, c] = X[:, i] / (X[:, j] + 0.01)
                c += 1
        return np.concatenate((X, new_features), axis=1)


def create_pipeline(column_transformer, classifier, feature_selector='passthrough'):
    return Pipeline([('column_transformer', column_transformer),
                     ('feature_selector', feature_selector),
                     ('classifier', classifier)])


def find_best_features(column_transformers, classifiers, feature_selectors):
    dfs_by_clf = {}
    for clf in classifiers:
        dfs_by_clf[clf] = pd.DataFrame(data=np.zeros((len(feature_selectors), len(column_transformers))),
                                       index=feature_selectors.keys(),
                                       columns=column_transformers.keys())
        for ct in column_transformers:
            for fs in feature_selectors:
                cv_score = cross_val_score(create_pipeline(column_transformers[ct],
                                                           classifiers[clf],
                                                           feature_selectors[fs]),
                                           X,
                                           y,
                                           cv=RepeatedStratifiedKFold(n_splits=5,
                                                                      random_state=0),
                                           scoring='roc_auc')
                dfs_by_clf[clf].loc[fs, ct] = cv_score.mean()
    return dfs_by_clf


'''
def get_x_y(df: pd.DataFrame):
    X = df.drop(['BAD_CLIENT'], axis=1)
    y = df['BAD_CLIENT']
    return X, y


def get_holdout_data(X: pd.DataFrame, y: pd.Series):
    X_train, y_train, X_holdout, y_holdout = train_test_split(X, y, random_state=0)
    return X_train, y_train, X_holdout, y_holdout


# def get_cv_splits(n_splits=5, n_repeats=10):
'''

df = pd.read_csv(os.path.join(DATA_PATH, 'cleaned_data.csv'))
num_vars = [var for var in df.columns.values if df[var].dtype != 'object' and var != 'BAD_CLIENT']
cat_vars = [var for var in df.columns.values if df[var].dtype == 'object']
cat_vars_with_ot_dropped = [var for var in cat_vars if var != 'OCCUPATION_TYPE']


def create_column_transformers():

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

    ct_without_occupation_type = ColumnTransformer([('cat_transformation', categorical_transformation_5, cat_vars_with_ot_dropped),
                                                    ('num_transformation', numerical_transformation_5, num_vars)])

    return ct_simple_transformer,  ct_with_poly_features, ct_with_ratio_features, ct_with_ratio_and_poly_features, ct_without_occupation_type

X = df.drop(['BAD_CLIENT'], axis=1)
y = df['BAD_CLIENT']

column_transformers = {'simple_transformer': ct_simple_transformer,
                       'with_poly_features': ct_with_poly_features,
                       'with_ratio_features': ct_with_ratio_features,
                       'with_ratio_and_poly_features': ct_with_ratio_and_poly_features,
                       'without_occupation_type': ct_without_occupation_type
                       }
classifiers = {'logreg': LogisticRegression(random_state=0),
               'rf': RandomForestClassifier(random_state=0),
               'lgbmclf': LGBMClassifier(random_state=0)
               }
feature_selectors = {percentile: SelectPercentile(score_func=f_classif,
                                                  percentile=percentile) for percentile in range(10, 101, 10)}


def features_transformation_info(df, classifier):
    ft_info = pd.DataFrame(data=np.zeros((len(feature_selectors), len(column_transformers))),
                           index=feature_selectors.keys(),
                           columns=column_transformers.keys())
    for ct in column_transformers:
        for fs in feature_selectors:
            cv_score = cross_val_score(create_pipeline(column_transformers[ct],
                                                        classifiers,
                                                        feature_selectors[fs]),
                                        X,
                                        y,
                                        cv=cv,
                                        scoring='roc_auc')
            ft_info.loc[fs, ct] = cv_score.mean()
    return ft_info
