from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
import pandas as pd

class PCA(PCA):
    '''Wrapper around sklearn.decomposition.PCA that gives a pd.DataFrame output.'''        
    
    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        self.original_feature_names = X.columns
        super().fit(X,y)
        return self
    
    def transform(self, X):
        vals = super().transform(X)
        
        columns = [f'pca_{n}' for n in range(vals.shape[1])]
        index = X.index
        
        X_ans = pd.DataFrame(vals, index=index, columns=columns)
        return X_ans
    
    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)
    
    def get_components(self):
        '''Wrapper around self.components_ that gives the original feature contributions
        to each component as a pd.DataFrame.'''
        
        vals = self.components_
        columns = [f'pca_{n}' for n in range(vals.shape[0])]
        components = pd.DataFrame(vals, index=columns, columns=self.original_feature_names)
        return components