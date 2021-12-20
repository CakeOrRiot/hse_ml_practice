import math

import pandera as pa


def validate_cabin(ticket):
    if isinstance(ticket, float) and math.isnan(ticket):
        return True
    tokens = ticket.split()
    return all(token[0].isalpha() and (token[1:].isnumeric() or len(token[1:]) == 0) for token in tokens)


train_schema = pa.DataFrameSchema({
    'PassengerId': pa.Column(int, checks=pa.Check.ge(0)),
    'Survived': pa.Column(int, checks=pa.Check.isin([0, 1])),
    'Pclass': pa.Column(int, checks=pa.Check.isin([1, 2, 3])),
    'Name': pa.Column(str, checks=pa.Check.str_length(min_value=1)),
    'Sex': pa.Column(str, checks=pa.Check.isin(['male', 'female'])),
    'Age': pa.Column(float, checks=pa.Check.ge(0), nullable=True),
    'SibSp': pa.Column(int, checks=pa.Check.ge(0)),
    'Parch': pa.Column(int, checks=pa.Check.ge(0)),
    'Ticket': pa.Column(str),
    'Fare': pa.Column(float, checks=pa.Check.ge(0), nullable=True),
    'Cabin': pa.Column(str, checks=pa.Check(lambda x: all(validate_cabin(t) for t in x)), nullable=True),
    'Embarked': pa.Column(str, checks=pa.Check.isin(['C', 'Q', 'S']), nullable=True)
})

test_schema = train_schema.remove_columns(['Survived'])

all_schema = train_schema.remove_columns(['Survived']) \
    .add_columns({'Survived': pa.Column(float, checks=pa.Check.isin([0, 1]), nullable=True)})

out_schema = pa.DataFrameSchema({
    'Age': pa.Column(int, checks=pa.Check.ge(0)),
    'Deck_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Deck_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Deck_3': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Deck_4': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Embarked_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Embarked_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Embarked_3': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Family_Size_Grouped_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Family_Size_Grouped_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Family_Size_Grouped_3': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Family_Size_Grouped_4': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Fare': pa.Column(int, checks=pa.Check.ge(0)),
    'Is_Married': pa.Column(int, checks=pa.Check.isin([0, 1])),
    'Pclass_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Pclass_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Pclass_3': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Sex_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Sex_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Survival_Rate': pa.Column(float, checks=pa.Check.in_range(0, 1)),
    'Survival_Rate_NA': pa.Column(float, checks=pa.Check.in_range(0, 1)),
    'Ticket_Frequency': pa.Column(int, checks=pa.Check.ge(0)),
    'Title_1': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Title_2': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Title_3': pa.Column(float, checks=pa.Check.isin([0, 1])),
    'Title_4': pa.Column(float, checks=pa.Check.isin([0, 1])),
})
