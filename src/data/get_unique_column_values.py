import pandas as pd
import joblib
import click


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def get_unique_column_values(input_path: str, output_path: str) -> None:
    """
    Записывает словарь уникальных значений категориальных признаков в pkl-файл
    :param input_path: путь к очищенным данным
    :param output_path: путь к получаемому pkl-файлу
    """
    df = pd.read_csv(input_path)
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    unique_values = {col: sorted(df[col].unique()) for col in cat_cols}
    joblib.dump(unique_values, output_path)


if __name__ == '__main__':
    get_unique_column_values()
