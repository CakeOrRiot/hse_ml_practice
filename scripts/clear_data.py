import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from utils import concat_df


def clear_data(df_all):
    df_all_corr = df_all.corr().abs().unstack().sort_values(
        kind="quicksort", ascending=False).reset_index()
    df_all_corr.rename(columns={"level_0": "Feature 1",
                                "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)

    df_all_corr[df_all_corr['Feature 1'] == 'Age']
    df_all['Embarked'] = df_all['Embarked'].fillna('S')

    med_fare = df_all.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    df_all['Fare'] = df_all['Fare'].fillna(med_fare)

    df_all['Deck'] = df_all['Cabin'].apply(
        lambda s: s[0] if pd.notnull(s) else 'M')

    idx = df_all[df_all['Deck'] == 'T'].index
    df_all.loc[idx, 'Deck'] = 'A'

    df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
    df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
    df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

    df_all.drop(['Cabin'], inplace=True, axis=1)

    df_all['Fare'] = pd.qcut(df_all['Fare'], 13)
    df_all['Age'] = pd.qcut(df_all['Age'], 10)

    df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1

    family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large',
                  11: 'Large'}
    df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

    df_all['Ticket_Frequency'] = df_all.groupby(
        'Ticket')['Ticket'].transform('count')

    df_all['Title'] = df_all['Name'].str.split(
        ', ', expand=True)[1].str.split('.', expand=True)[0]
    df_all['Is_Married'] = 0
    df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

    df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                              'Miss/Mrs/Ms')
    df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                              'Dr/Military/Noble/Clergy')

    def extract_surname(data):
        families = []

        for i in range(len(data)):
            name = data.iloc[i]

            if '(' in name:
                name_no_bracket = name.split('(')[0]
            else:
                name_no_bracket = name

            family = name_no_bracket.split(',')[0]
            title = name_no_bracket.split(',')[1].strip().split(' ')[0]

            for c in string.punctuation:
                family = family.replace(c, '').strip()

            families.append(family)

        return families

    df_all['Family'] = extract_surname(df_all['Name'])
    df_train = df_all.loc[:890]
    df_test = df_all.loc[891:]
    dfs = [df_train, df_test]

    non_unique_families = [
        x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
    non_unique_tickets = [
        x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

    df_family_survival_rate = df_train.groupby(
        'Family')['Survived', 'Family', 'Family_Size'].median()
    df_ticket_survival_rate = df_train.groupby(
        'Ticket')['Survived', 'Ticket', 'Ticket_Frequency'].median()

    family_rates = {}
    ticket_rates = {}

    for i in range(len(df_family_survival_rate)):
        # Checking a family exists in both training and test set, and has members more than 1
        if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
            family_rates[df_family_survival_rate.index[i]
            ] = df_family_survival_rate.iloc[i, 0]

    for i in range(len(df_ticket_survival_rate)):
        # Checking a ticket exists in both training and test set, and has members more than 1
        if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[df_ticket_survival_rate.index[i]
            ] = df_ticket_survival_rate.iloc[i, 0]

    mean_survival_rate = np.mean(df_train['Survived'])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train['Family'][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train['Family'][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['Family'].iloc[i] in family_rates:
            test_family_survival_rate.append(
                family_rates[df_test['Family'].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)

    df_train['Family_Survival_Rate'] = train_family_survival_rate
    df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
    df_test['Family_Survival_Rate'] = test_family_survival_rate
    df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train['Ticket'][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['Ticket'].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(
                ticket_rates[df_test['Ticket'].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)

    df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
    df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
    df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
    df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

    for df in [df_train, df_test]:
        df['Survival_Rate'] = (df['Ticket_Survival_Rate'] +
                               df['Family_Survival_Rate']) / 2
        df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2

    non_numeric_features = ['Embarked', 'Sex', 'Deck',
                            'Title', 'Family_Size_Grouped', 'Age', 'Fare']

    for df in dfs:
        for feature in non_numeric_features:
            df[feature] = LabelEncoder().fit_transform(df[feature])

    cat_features = ['Pclass', 'Sex', 'Deck',
                    'Embarked', 'Title', 'Family_Size_Grouped']
    encoded_features = []

    for df in dfs:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(
                df[feature].values.reshape(-1, 1)).toarray()
            n = df[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

    df_all = concat_df(df_train, df_test)
    drop_cols = ['Deck', 'Embarked', 'Family', 'Family_Size', 'Family_Size_Grouped', 'Survived',
                 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title',
                 'Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA']
    df_all.drop(columns=drop_cols, inplace=True)
    print(df_all.columns)
    X_train = StandardScaler().fit_transform(df_train.drop(columns=drop_cols))
    y_train = df_train['Survived'].values
    X_test = StandardScaler().fit_transform(df_test.drop(columns=drop_cols))
    return X_train, y_train, X_test
