import pandas as pd

from schemas import *


@pa.check_io(out=train_schema)
def get_train():
    df_train = pd.read_csv('../data/raw/train.csv')
    df_train = train_schema(df_train)
    df_train.name = 'Training Set'
    return df_train


@pa.check_io(out=test_schema)
def get_test():
    df_test = pd.read_csv('../data/raw/test.csv')
    df_test = test_schema(df_test)
    df_test.name = 'Test Set'
    return df_test
