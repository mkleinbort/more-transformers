from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import pandas as pd

class StandardScaler(StandardScaler):
    '''Extension of sklearn.preprocessing.StandardScaler that retains column names and index'''
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        
        scaled_values = super().transform(X)
        columns = X.columns
        index = X.index
        X_scaled = pd.DataFrame(data=scaled_values, columns=columns, index=index)
        
        return X_scaled
    
class MinMaxScaler(MinMaxScaler):
    '''Extension of sklearn.preprocessing.MinMaxScaler that retains column names and index'''
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        
        scaled_values = super().transform(X)
        columns = X.columns
        index = X.index
        X_scaled = pd.DataFrame(data=scaled_values, columns=columns, index=index)
        
        return X_scaled
    
class QuantileTransformer(QuantileTransformer):
    '''Extension of sklearn.preprocessing.QuantileTransformer that retains column names and index'''
    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        
        scaled_values = super().transform(X)
        columns = X.columns
        index = X.index
        X_scaled = pd.DataFrame(data=scaled_values, columns=columns, index=index)
        
        return X_scaled
    
class GetDummies(BaseEstimator, TransformerMixin):
    '''Creates dummy columns in a way that is consistent between fit and transform'''
    
    def __init__(self, prefix=None, prefix_sep='_', dummy_na=False, 
                 columns=None, sparse=False, drop_first=False, dtype=None):
        '''Convert categorical variable into dummy/indicator variables.

        Parameters
        ----------
        data : DataFrame
            Data of which to get dummy indicators.

        prefix : str, list of str, or dict of str, default None
            String to append DataFrame column names.
            Pass a list with length equal to the number of columns
            when calling get_dummies on a DataFrame. Alternatively, `prefix`
            can be a dictionary mapping column names to prefixes.

        prefix_sep : str, default '_'
            If appending prefix, separator/delimiter to use. Or pass a
            list or dictionary as with `prefix`.

        dummy_na : bool, default False
            Add a column to indicate NaNs, if False NaNs are ignored.

        columns : list-like, default None
            Column names in the DataFrame to be encoded.
            If `columns` is None then all the columns with
            `object` or `category` dtype will be converted.

        sparse : bool, default False
            Whether the dummy-encoded columns should be backed by
            a :class:`SparseArray` (True) or a regular NumPy array (False).

        drop_first : bool, default False
            Whether to get k-1 dummies out of k categorical levels by removing the
            first level.

        dtype : dtype, default np.uint8
            Data type for new columns. Only a single dtype is allowed.
        '''

        self.prefix     = prefix
        self.prefix_sep = prefix_sep
        self.dummy_na   = dummy_na
        self.columns    = columns
        self.sparse     = sparse
        self.drop_first = drop_first
        self.dtype      = dtype
        
    def get_kwargs(self):
        '''Helper method that returns the kwargs to be passed to pd.get_dummies()'''
        kwargs = dict(prefix     = self.prefix,
                      prefix_sep = self.prefix_sep,
                      dummy_na   = self.dummy_na,
                      columns    = self.columns,
                      sparse     = self.sparse,
                      drop_first = self.drop_first,
                      dtype      = self.dtype)
        return kwargs
        
    def fit(self, X, y=None):
        '''Learns from X what dummy columns need to be created.'''
        kwargs = self.get_kwargs()
        self.output_columns = pd.get_dummies(X, **kwargs).columns
        self.new_columns = [col for col in self.output_columns if col not in X.columns]
        return self
    
    def transform(self, X):
        '''Applies pd.get_dummies(X, **kwargs) where the kwargs are as defined in the __init__'''
        kwargs = self.get_kwargs()

        X_dummies = pd.get_dummies(X, **kwargs)
        
        for col in self.new_columns:
            if col not in X_dummies.columns:
                X_dummies[col] = 0
                
        ans = X_dummies.loc[:, self.output_columns]
        return ans
        