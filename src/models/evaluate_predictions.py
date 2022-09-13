import click
import pandas as pd
import json
from sklearn.metrics import roc_auc_score


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('input_predictions_path', type=click.Path(exists=True))
@click.argument('output_metrics_path', type=click.Path())
def evaluate_predictions(input_data_path: str, input_predictions_path: str, output_metrics_path: str) -> None:

    df = pd.read_csv(input_data_path)
    y = df['BAD_CLIENT']

    predictions = pd.read_csv(input_predictions_path, header=None)

    metrics = {'roc_auc': roc_auc_score(y, predictions.iloc[:, 1])}
    with open(output_metrics_path, 'w') as f:
        json.dump(metrics, f)


if __name__ == '__main__':
    evaluate_predictions()

'''
python -m src.models.evaluate_predictions data/processed/test_dataset.csv reports/predictions.csv reports/metrics.csv
'''