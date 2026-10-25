import numpy as np
import pandas as pd

class StatisticalFeats:
      def __init__(self,df,covariates,time_col,group_col):
          super(StatisticalFeats,self).__init__()
          self.df=df
          self.covariates=covariates
          self.time_col=time_col
          self.group_col=group_col
    

      def rolling_window_feats(self,window_len):
          """
          Docstring for store_level_feats
          This function constructs (store x product) level features for demand forecasting.
          For example, rolling mean, std, and price lags (at 7/14) for daily data. 
        
          Args
          ------
          param window_len: int 
                 Accepts an integer value depicting window length for computing rolling mean/std
          """
          self.df=self.df.sort_values(by=self.time_col,ascending=True)
          for col in self.covariates:
              self.df[f'{col}_Rolling_Mean_{window_len}']=self.df.groupby(self.group_col)[col].shift(1).rolling(window=window_len).mean()
              self.df[f'{col}_Rolling_Std_{window_len}']=self.df.groupby(self.group_col)[col].shift(1).rolling(window=window_len).std()
          return self.df       
      
      def create_lag_features(self,lag_list):
          """_summary_

          Args:
              lag_list (list): a list of lag instants 

          Returns:
              pd.DataFrame: a dataframe with lag features created for the covariates initialized in the StatisticalFeats class
          """
          for lag in lag_list:
              for col in self.covariates:
                  self.df[f'{col}_lag_{lag}']=self.df.groupby(self.group_col)[col].shift(lag)
          return self.df            