from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

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