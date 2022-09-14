import click
import pandas as pd

from src.models.optuna_objectives import optimize_logreg
from src.models.optuna_objectives import optimize_rf
from src.models.optuna_objectives import optimize_xgboost
from src.models.optuna_objectives import optimize_lgbm
from src.models.optuna_objectives import optimize_catboost


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_folder_path", type=click.Path())
def make_optuna_studies(input_data_filepath: str, output_folder_path: str) -> None:
    """
    Создает optuna studies для различных классификаторов, проводит оптимизацию и записывает результаты
    оптимизации в отдельный pkl-файл для каждого классификатора
    :param input_data_filepath: путь к данным
    :param output_folder_path: путь к папке, в которой сохраняются файлы с результатами оптимизации
    """

    df = pd.read_csv(input_data_filepath)

    optimize_logreg(output_folder_path, df)
    optimize_rf(output_folder_path, df)
    optimize_lgbm(output_folder_path, df)


if __name__ == "__main__":
    make_optuna_studies()
