import numpy as np
from xgboost import XGBClassifier
from credit_fraud_utils_data import *

def Model(x_train, y_train):

    
    neg_count = y_train.value_counts()[0]
    pos_count = y_train.value_counts()[1]
    model = XGBClassifier(
        n_estimators=200,
        max_depth=10,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg_count / pos_count,
        eval_metric='aucpr',
        random_state=42
    )

    model.fit(x_train, y_train)

    return model


if __name__ == '__main__':
    df = load_data(path_file=r"E:\ML projects & Tasks\creditCardFraudDetection\data\train.csv")
    x_train, y_train = feature_engineering(df)
    model = Model(x_train, y_train)


