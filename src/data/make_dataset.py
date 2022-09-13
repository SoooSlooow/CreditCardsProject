import pandas as pd
import click


def add_clients_id(application_df: pd.DataFrame) -> pd.DataFrame:
    unique_clients = application_df.drop(['ID'], axis=1).drop_duplicates().reset_index(drop=True)
    unique_clients['CLIENT_ID'] = unique_clients.index.values
    columns_for_merging = list(application_df.columns.values[1:])
    application_df_transformed = application_df.merge(unique_clients, how='inner', on=columns_for_merging)
    return application_df_transformed


def get_labels(application_df: pd.DataFrame, credit_df: pd.DataFrame) -> pd.Series:
    status_mapping = {'X': -2,
                      'C': -1,
                      '0': 0,
                      '1': 1,
                      '2': 2,
                      '3': 3,
                      '4': 4,
                      '5': 5
                      }
    credit_df_transformed = credit_df.copy(deep=True)
    credit_df_transformed['STATUS_INT'] = credit_df_transformed['STATUS'].map(status_mapping)
    credit_df_transformed = credit_df_transformed.merge(application_df[['ID', 'CLIENT_ID']],
                                                        on='ID',
                                                        how='inner')
    worst_status_by_client_id = credit_df_transformed.groupby('CLIENT_ID').max()['STATUS_INT']
    data_labels = (worst_status_by_client_id >= 3).astype(int)
    data_labels.name = 'BAD_CLIENT'
    return data_labels


def get_initial_df(application_df: pd.DataFrame, data_labels: pd.Series) -> pd.DataFrame:
    initial_preprocessed_df = application_df.merge(data_labels, how='inner', on='CLIENT_ID')
    subset = list(initial_preprocessed_df.columns.values)
    subset.remove('ID')
    initial_preprocessed_df = initial_preprocessed_df.drop(['ID'], axis=1).\
        drop_duplicates(subset=subset).reset_index(drop=True)
    return initial_preprocessed_df


@click.command()
@click.argument('input_filepath_app_data', type=click.Path(exists=True))
@click.argument('input_filepath_credit_data', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def make_dataset(input_filepath_app_data: str, input_filepath_credit_data: str, output_filepath: str) -> None:

    application_data = pd.read_csv(input_filepath_app_data)
    credit_data = pd.read_csv(input_filepath_credit_data)

    application_data_transformed = add_clients_id(application_data)
    labels = get_labels(application_data_transformed, credit_data)
    initial_preprocessed_data = get_initial_df(application_data_transformed, labels)
    initial_preprocessed_data.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    make_dataset()

'''
python -m src.data.make_dataset data/raw/application_record.csv data/raw/credit_record.csv data/interim/initial_dataset.csv
'''
