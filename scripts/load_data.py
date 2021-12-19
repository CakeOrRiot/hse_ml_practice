import pandas as pd

from utils import concat_df


def get_data():
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')
    df_all = concat_df(df_train, df_test)

    df_train.name = 'Training Set'
    df_test.name = 'Test Set'
    df_all.name = 'All Set'

    return df_train, df_test
