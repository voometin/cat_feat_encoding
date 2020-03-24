import category_encoders as ce
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


class Preprocessor:

    def __init__(self, label_encoding_columns=[], target_encoder_columns=[], ohe_columns=[], min_max_columns=[],
                 custom_transform={}, drop_columns=[]):
        self.label_encoding_columns = label_encoding_columns
        self.target_encoder_columns = target_encoder_columns
        self.ohe_columns = ohe_columns
        self.min_max_columns = min_max_columns
        self.custom_transform = custom_transform
        self.drop_columns = drop_columns
        self.isTrain = True
        self.target = None

        self.le = []
        self.ohe = []
        self.mm = []
        self.te = []

    def fit(self, X, target=None):
        X = X.copy()

        for col in self.label_encoding_columns:
            self.le.append(OrdinalEncoder())
            X.loc[~X[col].isna(), col] = self.le[-1].fit_transform(X.loc[~X[col].isna(), col].values.reshape(-1, 1))

        for col in self.custom_transform:
            if type(self.custom_transform[col]) is dict:
                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].replace(self.custom_transform[col])
            elif type(self.custom_transform[col]) is list:
                for sub_col, func in self.custom_transform[col]:
                    X[sub_col] = -1
                    X.loc[~X[col].isna(), sub_col] = X.loc[~X[col].isna(), col].apply(func)
            else:
                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].apply(self.custom_transform[col])

        for col in self.min_max_columns:
            self.mm.append(MinMaxScaler())
            X.loc[~X[col].isna(), col] = self.mm[-1].fit_transform(X.loc[~X[col].isna(), col].values.reshape(-1, 1))

        if self.target_encoder_columns:
            self.target = target
            for train_ind, val_ind in StratifiedKFold(shuffle=True, random_state=123).split(X, target):
                self.te.append(ce.TargetEncoder(cols=self.target_encoder_columns, handle_missing='return_nan'))
                self.te[-1].fit(X.loc[train_ind, self.target_encoder_columns],
                                X.loc[train_ind, 'target'].values.reshape(-1, 1))

            target_encoder = ce.TargetEncoder(cols=self.target_encoder_columns,
                                              handle_missing='return_nan')  # , smoothing=0.25))
            self.te.append(target_encoder)
            self.te[-1].fit(X[self.target_encoder_columns], target)

        return self

    def transform(self, X):
        X = X.copy()

        for ind, col in enumerate(self.label_encoding_columns):
            X.loc[~X[col].isin(list(self.le[ind].categories_[0])), col] = np.nan
            X.loc[~X[col].isna(), col] = self.le[ind].transform( X.loc[~X[col].isna(), col].values.reshape(-1, 1)) # .astype(int)

        for col in self.custom_transform:
            if type(self.custom_transform[col]) is dict:
                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].replace(self.custom_transform[col])
            elif type(self.custom_transform[col]) is list:
                for sub_col, func in self.custom_transform[col]:
                    X[sub_col] = -1
                    X.loc[~X[col].isna(), sub_col] = X.loc[~X[col].isna(), col].apply(func)
            else:
                X.loc[~X[col].isna(), col] = X.loc[~X[col].isna(), col].apply(self.custom_transform[col])

        for ind, col in enumerate(self.min_max_columns):
            X.loc[~X[col].isna(), col] = self.mm[ind].transform(X.loc[~X[col].isna(), col].values.reshape(-1, 1))

        if self.target_encoder_columns:
            if self.isTrain:  # train-val
                for ind, (train_ind, val_ind) in enumerate\
                            (StratifiedKFold(shuffle=True, random_state=123).split(X, self.target)):
                    X.loc[val_ind, self.target_encoder_columns] = self.te[ind].transform\
                        (X.loc[val_ind, self.target_encoder_columns])
            else:  # test
                X[self.target_encoder_columns] = self.te[-1].transform(X[self.target_encoder_columns])

        if self.drop_columns:
            X = X.drop(self.drop_columns, axis=1)
        return X


