import pytest

from scripts.load_data import get_test
from scripts.load_data import get_train


@pytest.fixture
def data_load():
    train = get_train('data/raw/train.csv')
    test = get_test('data/raw/test.csv')
    return train, test


def test_load(data_load):
    assert len(data_load) == 2


def test_columns(data_load):
    train, test = data_load
    assert 'Name' in train and 'Name' in test
    expected = {'PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                      'Embarked','Survived'}
    assert set(train.columns) == expected
    assert set(test.columns).issubset(expected)
