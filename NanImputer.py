class NanImputer:

    def __init__(self, mode: str):
        self.mode = mode  # ('ohe',) or ('fillna', -1)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if self.mode[0] == 'fillna':
            X.fillna(self.mode[1], inplace=True)
        elif self.mode[0] == 'ohe':
            nan_columns = X.isna().sum()
            nan_columns = nan_columns[nan_columns > 0].index.values
            for column in nan_columns:
                # X[f'{column}_isNaN'] = X[column].isna() * 1
                # X.loc[X[column].isna(), column] = X.loc[~X[column].isna(), column].value_counts().index.values.mean()
                X.loc[X[column].isna(), column] = X.loc[~X[column].isna(), column].values.mean()
                X[column] = X[column].astype(float)
        return X
