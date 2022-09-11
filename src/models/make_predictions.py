import click
import joblib
import pandas as pd
import numpy as np


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('input_model_path', type=click.Path(exists=True))
@click.argument('output_predictions_path', type=click.Path())
def make_predictions(input_data_path, input_model_path, output_predictions_path):
    df = pd.read_csv(input_data_path)
    X = df.drop(['BAD_CLIENT'], axis=1, errors='ignore')

    model = joblib.load(input_model_path)

    predictions = model.predict_proba(X)

    np.savetxt(output_predictions_path, predictions, delimiter=',')


if __name__ == '__main__':
    make_predictions()

'''
python -m src.models.make_predictions data/processed/test_dataset.csv models/final_model.pkl reports/predictions.csv
'''