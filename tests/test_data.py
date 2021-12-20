from scripts.load_data import get_test
from scripts.load_data import get_train


def test_test_data_load():
    test = get_test('data/raw/test.csv')
    assert len(test.columns) > 0


def test_train_data_load():
    train = get_train('data/raw/train.csv')
    assert len(train.columns) > 0
