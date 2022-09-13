import click
import pandas as pd
from sklearn.model_selection import ShuffleSplit


@click.command()
@click.argument('input_data_path', type=click.Path(exists=True))
@click.argument('output_train_data_path', type=click.Path())
@click.argument('output_test_data_path', type=click.Path())
def get_train_test(input_data_path: str, output_train_data_path: str, output_test_data_path: str) -> None:
    df = pd.read_csv(input_data_path)
    train_indices, test_indices = next(ShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(df))
    train_df = df.iloc[train_indices, :]
    test_df = df.iloc[test_indices, :]
    train_df.to_csv(output_train_data_path, index=False)
    test_df.to_csv(output_test_data_path, index=False)


if __name__ == '__main__':
    get_train_test()

'''
python -m src.data.get_train_test data/interim/cleaned_dataset.csv data/processed/train_dataset.csv data/processed/test_dataset.csv
'''