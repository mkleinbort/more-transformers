import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    '''Class to very flexibly select columns of a pandas DataFrame.'''
    def __init__(self, columns=None):
        '''Select columns of a pd.DataFrame according to input "columns"
        
        Note that columns can be None (i.e. select all columns),
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
                
        Note:
        -----
        I prefer "numeric" to "number", so this class has an exception where 
        ColumnSelector('numeric') will select columns as per X.select_dtypes('number')
        '''
        
        if columns is 'numeric':
            columns = 'number'
            
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
            
class RowSelector(BaseEstimator, TransformerMixin):
    '''Class to very flexibly select rows of a pandas DataFrame.'''
    
    def __init__(self, rows=None):
        '''Select rows of a pd.DataFrame according to input "rows"
        
        Note that rows can be None (i.e. select all rows),
        a list (i.e. select these rows as per the index),  
        or even a callable that will be passed to X.loc[rows,:].
        
        Note:
        -----
        My intended usage of this class is to exclude certain rows during training.
        
        So, for example, you could have rows be the callable
        
            lambda row: row['Col1'].between(-2, 2)
            
        to exclude any row with values in Col1 outside 2-std from the mean 
        (assuming normalized data)
        
        Parameters:
        ----------
        rows: {None, list, string, or callable} (default=None)
            - If None all rows are selected, as per
            
                X.loc[:, :]
            
            - If list then the rows in the given list are selected, as per
                
                X.loc[rows, :]
                
            - If callable then rows are selected as per 
            
                X.loc[rows, :]
                
        Example:
        --------
        
        model = Pipeline([
            ('exclude_123', RowSelector(lambda row: ~row.index.isin([1,2,3]))),
        ])
        
        model = Pipeline([
            ('first_100_rows', RowSelector(list(range(100)))),
            ('exclude_2_8_and_20', RowSelector(lambda row: ~row.index.isin([2,8,20]))),
            ('where_even_index', RowSelector(lambda row: row.index%2==0))
        ])
        
        '''
        self.rows = rows
        
    def fit(self, X, y=None):
        '''Nothing happens here'''
        return self

    def transform(self, X):
        '''Applies the selection of columns as per the documentation in __init__.'''
        
        assert isinstance(X, pd.DataFrame), f'X is expected to be a pd.DataFrame, but is of type {type(X)}'
        
        if self.rows is None:
            return X
        elif isinstance(self.rows, list) or isinstance(self.rows, tuple) or isinstance(self.rows, np.ndarray):
            return X.loc[self.rows]
        elif callable(self.rows):
            return X.loc[self.rows]
        else:
            raise ValueError(f'Could not interpret {self.rows} as a way to select rows.')
            
class ApplyFunction(BaseEstimator, TransformerMixin):
    def __init__(self, func=None, kwargs={}):
        assert callable(func) or func is None, 'func needs to be callable or None'
        self.func = func
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        '''Applies the function to X'''
        assert isinstance(X, pd.DataFrame)
        if self.func is None:
            ans = X
        else:
            ans = X.pipe(self.func, **self.kwargs)
        return ans
    
class PandasMethod(BaseEstimator, TransformerMixin):
    
    method_list = [func for func in dir(pd.DataFrame) if 
               callable(getattr(pd.DataFrame, func)) and
               not func.startswith('_')]
    
    def __init__(self, method=None, kwargs={}):
        assert method in self.method_list or method is None, f'pd.DataFrames do not have the method {method}'
        self.method = method
        self.kwargs = kwargs
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.method is None:
            ans = X
        else:
            ans = getattr(X,self.method)(**self.kwargs)
        return ans