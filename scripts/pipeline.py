import pandas as pd

import load_data
from clear_data import DataProcessor
from model import Model


def pipeline():
    train, test = load_data.get_train(), load_data.get_test()

    processor = DataProcessor(train, test)
    X_train, y_train, X_test = processor.get_data()

    model = Model()
    pred = model.fit_predict(X_train, y_train, X_test)
    result = pd.DataFrame(data=pred)
    result.to_csv('../data/submission.csv')
    return result


pipeline()
