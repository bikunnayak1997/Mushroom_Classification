import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        

    def get_data_transformer_object(self):
        try:
            logging.info('Data Transformation initiated')
            categorical_columns = ['bruises', 'gill-spacing', 'gill-size', 'gill-color','stalk-root', 'ring-type', 'spore-print-color']

            cat_pipeline = Pipeline(
                steps=[
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                    
                ]
            )
            logging.info(f'categorical columns:{categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ('cat_pipelines',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train DataFrame head:\n{train_df.head().to_string()}')
            logging.info(f'Test DataFrame head:\n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'class'
      

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on train and test dataframe')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            logging.info('applying label encode to target data')
            
            label_coder= LabelEncoder()
            train_arr = label_coder.fit_transform(target_feature_train_df)
            test_arr = label_coder.transform(target_feature_test_df)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')
            return (
                input_feature_test_df,
                train_arr,
                input_feature_test_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            logging.info('Error has occured ')
            raise CustomException(e, sys)
