import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet

class ElasticNetModel:
      def __init__(self,alpha,l1_penalty):
          self.alpha=alpha
          self.l1_penaty=l1_penalty
          self.random_state=56
          self.model=ElasticNet(alpha=self.alpha,l1_ratio=self.l1_penaty,
                                random_state=self.random_state)

      def train(self,X,y):
          if X.isnull().values.any():
             combined_df=pd.concat([X,y],axis=1)
             combined_df=combined_df.dropna()
             X=combined_df.drop(columns=y.name)
             y=combined_df[y.name]  
          assert X.shape[0]==y.shape[0]     
          self.model.fit(X,y)
          return self.model
       
      def predict(self,X):
          return self.model.predict(X)              