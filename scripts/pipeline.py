import load_data
from clear_data import clear_data
from model import Model
from utils import concat_df


def run():
    train, test = load_data.get_data()
    X_train, y_train, X_test = clear_data(concat_df(train, test))
    model = Model()
    pred = model.fit_predict(X_train, y_train, X_test)
    return pred


run()
