import pandas as pd
import click


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def clean_dataset(input_filepath, output_filepath):

    df = pd.read_csv(input_filepath)

    df.loc[df['DAYS_EMPLOYED'] > 0, 'OCCUPATION_TYPE'] = df.loc[df['DAYS_EMPLOYED'] > 0, 'OCCUPATION_TYPE'].\
        fillna('Unemployed')
    df.loc[df['DAYS_EMPLOYED'] <= 0, 'OCCUPATION_TYPE'] = df.loc[df['DAYS_EMPLOYED'] <= 0, 'OCCUPATION_TYPE'].\
        fillna('Other Occupations')

    df.loc[df['DAYS_EMPLOYED'] > 0, 'DAYS_EMPLOYED'] = 1

    flag_cols = ['FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL']
    df[flag_cols] = df[flag_cols].applymap(lambda x: 'Y' if x == 1 else 'N')

    df['YEARS_BIRTH'] = -df['DAYS_BIRTH'] / 365.2425
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365.2425
    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'CLIENT_ID'], axis=1)

    df['CNT_FAM_MEMBERS'] = df['CNT_FAM_MEMBERS'].astype('int64')

    cols = ['CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'AMT_INCOME_TOTAL', 'YEARS_BIRTH',
            'YEARS_EMPLOYED', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
            'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE', 'FLAG_WORK_PHONE', 'FLAG_PHONE',
            'FLAG_EMAIL', 'OCCUPATION_TYPE', 'BAD_CLIENT'
            ]
    df = df[cols]

    df.to_csv(output_filepath, index=False)


if __name__ == '__main__':
    clean_dataset()

# python -m src.data.clean_dataset data/interim/initial_dataset.csv data/interim/cleaned_dataset.csv
