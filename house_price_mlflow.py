import os
import sys
import warnings
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    r2 = r2_score(true, pred)
    return mae, mse, r2

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    np.random.seed(1)

    try:
        df = pd.read_csv('house_price.csv')
    except Exception as e:
        logger.exception('file not present at the given path or incorrect file name')

    X = df.drop(['id', 'date', 'price'],axis=1)
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    # l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    mlflow.set_experiment("boosting_houseprice_model")
    experiment = mlflow.get_experiment_by_name("boosting_houseprice_model")

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        gbr = GradientBoostingRegressor(n_estimators = estimators, max_depth = depth, learning_rate = lr, random_state=1)
        gbr.fit(X_train, y_train)

        y_pred = gbr.predict(X_test)

        mae, mse, r2 = eval_metrics(y_test, y_pred)
        print(f'Gradient Boosting model with estimators = {estimators}, max_depth = {depth} and learning rate = {lr}')
        print(f'MAE = {mae}')
        print(f'MSE = {mse}')
        print(f'R2 = {r2}')

        mlflow.log_param('estimators', estimators)
        mlflow.log_param('depth', depth)
        mlflow.log_param('lr', lr)
        mlflow.log_metric('MAE', mae)
        mlflow.log_metric('MSE', mse)
        mlflow.log_metric('R2', r2)

        predictions = gbr.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':

            mlflow.sklearn.log_model(
                gbr, 'boosting_houseprice_model', registered_model_name='BoostingPriceModel', signature=signature
            )
        else:

            mlflow.sklearn.log_model(gbr, 'boosting_houseprice_model', signature=signature)