import pandas as pd
import click
from src.models.model_selection_utils import optimize_logreg
from src.models.model_selection_utils import optimize_rf
from src.models.model_selection_utils import optimize_xgboost
from src.models.model_selection_utils import optimize_lgbm
from src.models.model_selection_utils import optimize_catboost


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_folder_path', type=click.Path())
def make_optuna_studies(input_filepath, output_folder_path):

    df = pd.read_csv(input_filepath)

    #optimize_logreg(output_folder_path, df)
    #optimize_rf(output_folder_path, df)
    #optimize_xgboost(output_folder_path, df)
    optimize_lgbm(output_folder_path, df)
    #optimize_catboost(output_folder_path, df)


if __name__ == '__main__':
    make_optuna_studies()

'''
python -m src.models.make_optuna_studies data/interim/cleaned_dataset.csv models/optuna_studies
'''
