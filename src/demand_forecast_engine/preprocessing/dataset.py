import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from pathlib import Path
import yaml

class DataSetLoader:
      def __init__(self,file_path,config_path="config/base.yaml"):
          super(DataSetLoader,self).__init__()
          project_root = Path.cwd()
          self.file_path = project_root/file_path
          self.config_path = project_root/config_path
          self.scaler=MinMaxScaler()
          with open(self.config_path) as file:
               self.config = yaml.safe_load(file)
      
      def read_data(self):
         if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

         if self.file_path.suffix==".csv":
            return pd.read_csv(self.file_path)

         if self.file_path.suffix in [".xlsx", ".xls"]:
            return pd.read_excel(self.file_path)

         raise ValueError(f"Unsupported file type: {self.file_path}")
             

      def cleandata(self,df):
          # Check for null values
          if df.isnull().any(axis=1).sum()>int(0.8*len(df)):
             raise ValueError("More than 80% of rows contain missing values")

          elif df.isnull().any(axis=1).sum()<int(0.8*len(df)):
               cols_to_impute=[cols for cols in df.columns if df[cols].dtype in ["float","int"]]
               numeric_df=df[cols_to_impute]
               numeric_scaled_df=self.scaler.fit_transform(numeric_df) 
               imputer=KNNImputer(n_neighbors=3)
               imputed_values=imputer.fit_transform(numeric_scaled_df)
               df[cols_to_impute]=self.scaler.inverse_transform(imputed_values)
               df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y",errors="coerce")
               df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
          #Check outliers
          return df

class CreateTabularData(DataSetLoader):
    """
    Handles the transformation of raw DataFrames into model-ready tabular datasets.
    
    This class manages data scaling, categorical encoding, and time-based 
    splitting for both deep learning models (requiring scaling) and tree-based 
    models (XGBoost, Random Forest, LightGBM).

    Attributes:
        df (pd.DataFrame): The main dataframe containing features and targets.
        train_cut_off_date (str/datetime): Date used to split training and testing sets.
        model_type (str): Type of model being used (e.g., 'XGBoost', 'LightGBM').
        scale_target (bool): Whether the target column should be normalized.
        exclude_cols (list): Columns to be ignored during numerical scaling.
    """

    def __init__(self, df, model_type, file_path, config_path):
        """
        Initializes CreateTabularData with data configuration and model parameters.

        Args:
            df (pd.DataFrame): Raw dataframe post feature engineering.
            model_type (str): The specific ML model type.
            file_path (str): Path to the data source.
            config_path (str): Path to the JSON/YAML configuration file.
        """
        super(CreateTabularData, self).__init__(file_path, config_path)
        self.df = df
        self.train_cut_off_date = self.config["data"]["train_test_split"]["train_cut_off_date"]
        self.model_type = model_type
        if self.model_type in ["XGBoost", "RF", "LightGBM"]:
            self.scale_target = False
        else:
            self.scale_target = True
        self.exclude_cols = self.config["features"]["exclude"]

    def scale_data(self, df):
        """
        Applies numerical scaling and categorical encoding to the dataframe.
        
        Numerical features are scaled using MinMaxScaler per hierarchy group.
        Categorical features are transformed using Label Encoding. Finally,
        the data is filtered based on the specific category/region criteria
        defined in the config.

        Args:
            df (pd.DataFrame): Dataframe to be scaled.

        Returns:
            pd.DataFrame: Scaled and filtered dataframe.
        """
        exclude_cols = self.exclude_cols
        numeric_cols = df.select_dtypes(include=[np.number]).columns.difference(exclude_cols)
        
        self.scaler = MinMaxScaler()
        scaled_df = df.copy() 
        
        for col in numeric_cols:
            scaled_df[col] = df.groupby(self.config["data"]["hierarchy"])[col].transform(
                lambda x: self.scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
            )
        
        categorical_cols = self.config["features"]["categorical"]
        for col in categorical_cols:
            le = LabelEncoder()
            scaled_df[col] = le.fit_transform(df[col].astype('str'))  
      
        if self.scale_target:
            scaled_df = self.target_scaler(scaled_df)  
          
        scaled_df = scaled_df[(scaled_df["Category"] == self.config["data"]["filter"]['Category']) &
                              (scaled_df["Region"] == self.config["data"]["filter"]['Region']) &
                              (scaled_df["Product ID"] == self.config["data"]["filter"]['Product ID']) &
                              (scaled_df["Store ID"] == self.config["data"]["filter"]['Store ID'])]
    
        return scaled_df

    def target_scaler(self, df):
        """
        Normalizes the target variable using MinMaxScaler.
        
        Scaling is performed independently for each group defined in the 
        hierarchy configuration to maintain relative trends.

        Args:
            df (pd.DataFrame): Dataframe containing the target column.

        Returns:
            pd.DataFrame: Dataframe with the scaled target column.
        """
        self.target = self.config["data"]["target_col"]
        scaler = MinMaxScaler()
        df[self.target] = df.groupby(self.config["data"]["hierarchy"])[self.target].transform(
            lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel()
        )
        return df

    def train_test_data(self, df):
        """
        Splits the data into training and testing sets based on a cutoff date.
        
        The method handles two distinct pipelines:
        1. Deep Learning: Scales the entire dataset before splitting.
        2. Tree-Based (XGBoost/RF/LGBM): Filters data first, applies Label Encoding, 
           and drops hierarchy/date columns before returning raw features.

        Args:
            df (pd.DataFrame): Raw dataframe post feature engineering.

        Returns:
            tuple: Contains (X_train, y_train, X_test, y_test) as DataFrames/Series.
        """
        if self.model_type not in ["XGBoost", "RF", "LightGBM"]:
            # Logic for models requiring scaling
            train_df = df[df["Date"] < self.train_cut_off_date]
            test_df = df[df["Date"] > self.train_cut_off_date]
            
            scaled_train_df = self.scale_data(train_df)
            scaled_test_df = self.scale_data(test_df)

            X_train, y_train = scaled_train_df.drop(columns=[self.config["data"]["target_col"]]), \
                               scaled_train_df[self.config["data"]["target_col"]]
          
            X_test, y_test = scaled_test_df.drop(columns=[self.config["data"]["target_col"]]), \
                             scaled_test_df[self.config["data"]["target_col"]]
          
        else:
            # Logic for tree-based models
            df = df.groupby(self.config["data"]["hierarchy"]).apply(lambda x: x.reset_index(drop=True))
            
            df = df[(df["Category"] == self.config["data"]["filter"]['Category']) &
                    (df["Region"] == self.config["data"]["filter"]['Region']) &
                    (df["Product ID"] == self.config["data"]["filter"]['Product ID']) &
                    (df["Store ID"] == self.config["data"]["filter"]['Store ID'])]
    
            categorical_cols = self.config["features"]["categorical"]

            def label_encode_df(df, categorical_cols):
                df = df.copy()
                encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le
                return df, encoders
               
            train_df, encoders = label_encode_df(df[df["Date"] < self.train_cut_off_date], categorical_cols)
            test_df = df[df["Date"] > self.train_cut_off_date].copy()

            for col, le in encoders.items():
                test_df[col] = le.transform(test_df[col].astype(str))
              
            cols_to_drop = self.config["data"]["hierarchy"] + ["Date"]
            
            train_df = train_df.drop(columns=cols_to_drop, axis=1)
            test_df = test_df.drop(columns=cols_to_drop, axis=1)
            
            X_train, y_train = train_df.drop(columns=[self.config["data"]["target_col"]]), \
                               train_df[self.config["data"]["target_col"]]
          
            X_test, y_test = test_df.drop(columns=[self.config["data"]["target_col"]]), \
                             test_df[self.config["data"]["target_col"]]
           
        return (X_train, y_train, X_test, y_test)