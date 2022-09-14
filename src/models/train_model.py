import click
import joblib
from lightgbm import LGBMClassifier
import pandas as pd

from src.utils import create_column_transformers, create_pipeline


@click.command()
@click.argument("input_data_path", type=click.Path(exists=True))
@click.argument("output_model_path", type=click.Path())
def train_model(input_data_path: str, output_model_path: str) -> None:
    """
    Применяет преобразование признаков к очищенным данных и обучает модель
    :param input_data_path: путь к очищенным данным
    :param output_model_path: путь к pkl-файлу с получаемой обученной моделью
    """
    df = pd.read_csv(input_data_path)
    X = df.drop(["BAD_CLIENT"], axis=1)
    y = df["BAD_CLIENT"]

    column_transformer = create_column_transformers(X)["without_occupation_type"]
    model_params = {
        "lambda_l2": 31.142026352479732,
        "max_depth": 40,
        "n_estimators": 320,
        "learning_rate": 0.09130234251787786,
    }
    model = LGBMClassifier(random_state=0, **model_params)

    pipeline = create_pipeline(column_transformer, model)
    pipeline.fit(X, y)

    joblib.dump(pipeline, output_model_path)


if __name__ == "__main__":
    train_model()

"""
python -m src.models.train_model data/processed/train_dataset.csv models/final_model.pkl
"""
