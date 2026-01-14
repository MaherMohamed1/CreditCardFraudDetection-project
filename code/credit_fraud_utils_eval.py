import pandas as pd
from credit_fraud_utils_data import *
from credit_fraud_utils_train import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    precision_recall_curve, roc_curve, average_precision_score
import matplotlib.pyplot as plt


def evaluation(model, x_train, y_train,x_val, y_val, x_test, y_test):

    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    y_test_pred = model.predict(x_test)

    y_pred_proba = model.predict_proba(x_test)[:, 1]

    train_f1_score = f1_score(y_train, y_train_pred)
    train_precision  = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)

    val_f1_score = f1_score(y_val, y_val_pred)
    val_precision  = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)

    test_f1_score = f1_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)

    print(f"Train:\n    F1: {train_f1_score:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")
    print(f"Val:\n    F1: {val_f1_score:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f}")
    print(f"Test:\n    F1: {test_f1_score:.4f} | Precision: {test_precision:.4f} | Recall: {test_recall:.4f}")

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision_recall_curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train_df = load_data(path_file=r"E:\ML projects & Tasks\creditCardFraudDetection\data\train.csv")
    val_df = load_data(path_file=r'E:\ML projects & Tasks\creditCardFraudDetection\data\val.csv')
    test_df = load_data(path_file=r'E:\ML projects & Tasks\creditCardFraudDetection\data\test.csv')

    x_train, y_train = feature_engineering(df=train_df)
    x_val, y_val = feature_engineering(df=val_df)
    x_test, y_test = feature_engineering(df=test_df)

    model = Model(x_train, y_train)
    evaluation(model, x_train, y_train,x_val, y_val, x_test, y_test)