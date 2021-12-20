import pandas as pd

from scripts import load_data
from scripts.clear_data import DataProcessor
from scripts.model import Model


class Pipeline:
    def __init__(self, path):
        self.path = path

    def run(self):
        train, test = load_data.get_train(f'{self.path}/raw/train.csv'), load_data.get_test(f'{self.path}/raw/test.csv')

        processor = DataProcessor(train, test, self.path)
        X_train, y_train, X_test = processor.get_data()

        model = Model()
        pred = model.fit_predict(X_train, y_train, X_test)
        result = pd.DataFrame(data=pred)
        result.to_csv(f'{self.path}/submission.csv')
        return result


if __name__ == '__main__':
    Pipeline('../data').run()
