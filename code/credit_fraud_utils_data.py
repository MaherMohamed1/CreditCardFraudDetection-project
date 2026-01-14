import numpy as np
import pandas as pd

def load_data(path_file):
    df = pd.read_csv(path_file)

    return df

def feature_engineering(df):

    df['Hour'] = (df['Time'] % 86400) / 3600
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day'] = (df['Time'] // 86400) % 7

    # Weekend flag (assuming Day 0 is Monday)
    df['Is_weekend'] = (df['Day'] >= 5).astype(int)


    x = df.drop(columns=['Class', 'Time'])
    y = df['Class']

    return x, y

if __name__ == '__main__':
    train_df = load_data(path_file=r'E:\ML projects & Tasks\creditCardFraudDetection\data\train.csv')
    val_df = load_data(path_file=r'E:\ML projects & Tasks\creditCardFraudDetection\data\val.csv')
    test_df = load_data(path_file=r'E:\ML projects & Tasks\creditCardFraudDetection\data\test.csv')

    x_train, y_train = feature_engineering(df=train_df)
    x_val, y_val = feature_engineering(df=val_df)
    x_test, y_test = feature_engineering(df=test_df)

