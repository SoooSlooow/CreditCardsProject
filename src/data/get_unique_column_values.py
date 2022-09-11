import pandas as pd
import joblib
import click


@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
def get_unique_column_values(input_path, output_path):
    df = pd.read_csv(input_path)
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    unique_values = {col: sorted(df[col].unique()) for col in cat_cols}
    joblib.dump(unique_values, output_path)


if __name__ == '__main__':
    get_unique_column_values()


'''python -m src.data.get_unique_column_values data/interim/cleaned_dataset.csv reports/unique_column_values.pkl'''