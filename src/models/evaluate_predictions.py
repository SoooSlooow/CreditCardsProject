import json

import click
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score


@click.command()
@click.argument("input_data_path", type=click.Path(exists=True))
@click.argument("input_predictions_path", type=click.Path(exists=True))
@click.argument("output_metrics_path", type=click.Path())
def evaluate_predictions(
    input_data_path: str, input_predictions_path: str, output_metrics_path: str
) -> None:
    """
    Генерирует набор метрик на основе истинных и предсказанных меток класса и записывает их в json-файл
    :param input_data_path: путь к данным
    :param input_predictions_path: путь к предсказанным меткам
    :param output_metrics_path: путь к получаемым метрикам
    """
    df = pd.read_csv(input_data_path)
    y = df["BAD_CLIENT"]

    predictions = pd.read_csv(input_predictions_path)

    metrics = {
        "roc_auc": roc_auc_score(y, predictions["proba_1"]),
        "precision": precision_score(y, predictions["label"]),
        "recall": recall_score(y, predictions["label"]),
    }
    with open(output_metrics_path, "w") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    evaluate_predictions()

"""
python -m src.models.evaluate_predictions data/processed/test_dataset.csv reports/predictions.csv reports/metrics.csv
"""
