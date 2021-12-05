import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


class Model:
    def __init__(self, seed=42):
        self.seed = seed
        self.N = 5

        self.model = RandomForestClassifier(criterion='gini',
                                            n_estimators=1750,
                                            max_depth=7,
                                            min_samples_split=6,
                                            min_samples_leaf=6,
                                            max_features='auto',
                                            oob_score=True,
                                            random_state=self.seed,
                                            n_jobs=-1,
                                            verbose=1)

        self.model = self.model

    def fit_predict(self, X_train, y_train, X_test):
        # `StratifiedKFold` is used for stratifying the target variable. The folds are made by preserving the percentage of samples for each class in target variable (`Survived`).

        oob = 0
        probs = pd.DataFrame(np.zeros((len(X_test), self.N * 2)),
                             columns=['Fold_{}_Prob_{}'.format(i, j) for i in range(1, self.N + 1) for j in range(2)])

        # index=df_all.columns)
        fprs, tprs, scores = [], [], []

        skf = StratifiedKFold(n_splits=self.N, random_state=self.N, shuffle=True)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            print('Fold {}\n'.format(fold))

            # Fitting the model
            self.model.fit(X_train[trn_idx], y_train[trn_idx])

            # Computing Train AUC score
            trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx],
                                                         self.model.predict_proba(X_train[trn_idx])[:, 1])
            trn_auc_score = auc(trn_fpr, trn_tpr)
            # Computing Validation AUC score
            val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx],
                                                         self.model.predict_proba(X_train[val_idx])[:, 1])
            val_auc_score = auc(val_fpr, val_tpr)

            scores.append((trn_auc_score, val_auc_score))
            fprs.append(val_fpr)
            tprs.append(val_tpr)

            # X_test probabilities
            probs.loc[:, 'Fold_{}_Prob_0'.format(
                fold)] = self.model.predict_proba(X_test)[:, 0]
            probs.loc[:, 'Fold_{}_Prob_1'.format(
                fold)] = self.model.predict_proba(X_test)[:, 1]

            oob += self.model.oob_score_ / self.N
            # print('Fold {} OOB Score: {}\n'.format(fold, self.model.oob_score_))

        fprs, tprs, scores = [], [], []

        skf = StratifiedKFold(n_splits=self.N, random_state=self.N, shuffle=True)

        for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            print('Fold {}\n'.format(fold))

            # Computing Train AUC score
            trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx],
                                                         self.model.predict_proba(X_train[trn_idx])[:, 1])
            trn_auc_score = auc(trn_fpr, trn_tpr)
            # Computing Validation AUC score
            val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx],
                                                         self.model.predict_proba(X_train[val_idx])[:, 1])
            val_auc_score = auc(val_fpr, val_tpr)

            scores.append((trn_auc_score, val_auc_score))
            fprs.append(val_fpr)
            tprs.append(val_tpr)

            # X_test probabilities
            probs.loc[:, 'Fold_{}_Prob_0'.format(
                fold)] = self.model.predict_proba(X_test)[:, 0]
            probs.loc[:, 'Fold_{}_Prob_1'.format(
                fold)] = self.model.predict_proba(X_test)[:, 1]

        class_survived = [col for col in probs.columns if col.endswith('Prob_1')]
        probs['1'] = probs[class_survived].sum(axis=1) / self.N
        probs['0'] = probs.drop(columns=class_survived).sum(axis=1) / self.N
        probs['pred'] = 0
        pos = probs[probs['1'] >= 0.5].index
        probs.loc[pos, 'pred'] = 1

        y_pred = probs['pred'].astype(int)
        return y_pred
