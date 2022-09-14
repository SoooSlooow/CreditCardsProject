import click
import joblib
import pandas as pd
import numpy as np


@click.command()
@click.argument("input_data_path", type=click.Path(exists=True))
@click.argument("input_model_path", type=click.Path(exists=True))
@click.argument("output_predictions_path", type=click.Path())
def make_predictions(
    input_data_path: str, input_model_path: str, output_predictions_path: str
) -> None:
    """
    Предсказывает значения меток в входных данных, используя подаваемую на вход модель.
    Предсказания записываются в csv-файл с тремя столбцами. В первые два столбца записываются вероятности
    отнесения объекта к классу 0 и 1 соответственно, в третий - предсказываемая метка объекта на основе
    выбранного порога вероятности.
    :param input_data_path: путь к данным
    :param input_model_path: путь к обученной модели
    :param output_predictions_path: путь к файлу с получаемыми предсказаниями
    """
    df = pd.read_csv(input_data_path)
    X = df.drop(["BAD_CLIENT"], axis=1, errors="ignore")

    model = joblib.load(input_model_path)
    probas = model.predict_proba(X)
    labels = (probas[:, 1] > 0.01).astype(int)
    predictions = pd.DataFrame(
        data=np.column_stack([probas, labels]), columns=["proba_0", "proba_1", "label"]
    )

    predictions.to_csv(output_predictions_path, index=False)


if __name__ == "__main__":
    make_predictions()

"""
python -m src.models.make_predictions data/processed/test_dataset.csv models/final_model.pkl reports/predictions.csv
"""
