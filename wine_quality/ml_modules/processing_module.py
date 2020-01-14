import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """
        Impute missing values:
        - Columns of dtype object are imputed with the most frequent value in column.
        - Columns of other types are imputed with mean of column.
        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
    
def add_predict_prob(est_dict,df):
    df.reset_index(inplace=True,drop=True)
    base_df = df
    for name, clf in est_dict.items():
        proba = clf.predict_proba(base_df)
        df_predict_proba = pd.DataFrame(proba, columns=[name+"_prob_label"+str(x) for x in range(len(proba[0]))])
        df = pd.merge(df, df_predict_proba,left_index=True, right_index=True)
    return df