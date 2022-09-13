import pandas as pd
import click


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def clean_dataset(input_filepath: str, output_filepath: str) -> None:
    """
    Производит очистку данных
    :param input_filepath: путь к изначальным данным
    :param output_filepath: путь к получаемым очищенным данным
    """

    df = pd.read_csv(input_filepath)

    df.loc[df['DAYS_EMPLOYED'] > 0, 'OCCUPATION_TYPE'] = df.loc[df['DAYS_EMPLOYED'] > 0, 'OCCUPATION_TYPE'].\
        fillna('Unemployed')
    df.loc[df['DAYS_EMPLOYED'] <= 0, 'OCCUPATION_TYPE'] = df.loc[df['DAYS_EMPLOYED'] <= 0, 'OCCUPATION_TYPE'].\
        fillna('Other Occupations')

    df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 1

    flag_cols = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                 'FLAG_MOBIL', 'FLAG_WORK_PHONE',
                 'FLAG_PHONE', 'FLAG_EMAIL']
    df[flag_cols] = df[flag_cols].applymap(lambda x: 'Yes' if x == 1 else 'No')

    df['YEARS_BIRTH'] = -df['DAYS_BIRTH'] / 365.2425
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365.2425
    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'CLIENT_ID'], axis=1)

    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype('int64')

    cols = [
        'YEARS_BIRTH', 'CODE_GENDER', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE',
        'YEARS_EMPLOYED', 'OCCUPATION_TYPE', 'NAME_EDUCATION_TYPE', 'CNT_FAM_MEMBERS',
        'CNT_CHILDREN', 'NAME_FAMILY_STATUS', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'NAME_HOUSING_TYPE', 'FLAG_PHONE', 'FLAG_WORK_PHONE', 'FLAG_EMAIL', 'BAD_CLIENT'
    ]
    df = df[cols]

    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    clean_dataset()

