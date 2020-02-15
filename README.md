# more-transformers

<img src="https://upload.wikimedia.org/wikipedia/commons/a/af/Clitheroe_Castle_wind_pipes.JPG" alt="Image of Pipes" width="200"/>
My own list of "extra" transformers in scikit-learn pipelines.

# Intro

When building scikit-learn pipelines I often feel I have to do a lot of my data preparation work outside the pipeline. Moreover, many scikit-learn transformers could be more beginer friendly if they returned pandas DataFrames instead of numpy arrays.

With that in mind, this library includes a few additional transformers that are mostly thin wrappers around scikit-learn. 

For example:

    from more_transformers.preprocessing import StandardScaler
    
behaves identically to `sklearn.preprocessing.StandardScaler` but returns a pandas DataFrame with the same column names and index values as the original.

As another example

    from more_transformers.decomposition import PCA
    
is the same as `from sklearn.decomposition import PCA` but retains the index and uses column names `pca_0`, `pca_1`,...,`pca_n`.


I've also added my own few helpers, mostly under `from more_transformers.common`. For example

    from more_transformers.preprocessing import GetDummies
    
is a transformer version of pd.get_dummies. One advantage is that if the test data is transformed to have the same columns as pd.get_dummies on the training data.

Also note 

    from more_transformers.common import ColumnSelector
    
allows for very flexible selection of columns in your pipeline. For example

    ColumnSelector() # Selects all columns
    ColumnSelector(['Age','Weight','Height']) # Selects these columns
    ColumnSelector('number') # Selects all integer or float columns
    ColumnSelector(lambda x: str(x).starts_with('x_'))  # Selects columns starting with 'x_'
    
    



