import numpy as np
import pandas as pd
from xgboost import XGBRegressor

class XGBoostModel:
      def __init__(self,l1_weight,l2_weight,objective):
          super(XGBoostModel,self).__init__()
          self.l1_weight=l1_weight
          self.l2_weight=l2_weight
          self.objective=objective
          self.random_state=42

      def train_xgb(self,X,y):
          if not isinstance(X, (pd.DataFrame, np.ndarray)):
           raise TypeError("X must be a pandas DataFrame or numpy array")

          if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError("y must be a pandas Series, numpy array")
   
          self.covariate_dimension=X.shape[1]
          print(f'Intializing training sequence....')
          self.model=XGBRegressor(objective=self.objective,reg_alpha=self.l1_weight,
                                  reg_lambda=self.l2_weight,random_state=self.random_state)
          self.model.fit(X,y)
          print(f'XGBoost for regression- Training completed')

          return self.model
      
      def forecast_func(self,X):
          assert X.shape[1]==self.covariate_dimension 
          if self.model is not None:
             forecast_vals=self.model.predict(X)
             print(f' Completed forecast for XGBoost')
          else:
             raise ValueError("Model is not trained")   
          return forecast_vals
      
         

             
           
         

          