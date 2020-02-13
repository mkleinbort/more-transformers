import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    '''Class to very flexibly select columns of a pandas DataFrame.'''
    def __init__(self, columns=None):
        ''' Select columns of a pd.DataFrame according to input "columns"
        
        Note that columns can be None (i.e. selectall columns),
        a list (i.e. select these columns),  
        a string (i.e. select appropiate dtypes as per X.select_dypes(include=[columns]))
        or even a callable that will be passed to X.loc[:, columns].
        
        Parameters
        ----------
        columns: {None, list, string, or callable} (default=None)
            - If None all columns are selected, as per
            
                X.loc[:, :]
            
            - If list then the columns in the given list are selected, as per
                
                X.loc[:, columns]
           
            - If string then columns are selected as per
                
                X.select_dtypes(include=[columns])
                
            - If callable then columns are selected as per 
            
                X.loc[:, columns]
        '''
        self.columns = columns
        
    def fit(self, X, y=None):
        '''Nothing happens here'''
        return self

    def transform(self, X):
        '''Applies the selection of columns as per the documentation in __init__.'''
        
        assert isinstance(X, pd.DataFrame), f'X is expected to be a pd.DataFrame, but is of type {type(X)}'
        
        if self.columns is None:
            return X
        elif isinstance(self.columns, list) or isinstance(self.columns, tuple) or isinstance(self.columns, np.ndarray):
            return X.loc[:, self.columns]
        elif isinstance(self.columns, str):
            return X.select_dtypes(include=[self.columns])
        elif callable(self.columns):
            return X.loc[:, self.columns]
        else:
            raise ValueError(f'Could not interpret {self.columns} as a way to select columns.')