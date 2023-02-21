import pandas as pd
import numpy as np


class ExtractLetterTransformer():
    # Extract fist letter of variable

    def __init__(self, variables):
        self.variables = variables
    
    def fit(self, X, y=None):
        return self

    def extractLetter(self, row):
        try:
            return row[0]
        except:
            pass
        
    def transform(self, X, y=None):
        # make a copy
        df = X.copy()
        df[self.variables] = df[self.variables].apply(self.extractLetter)
        return df
