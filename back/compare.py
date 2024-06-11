import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    model_forest  = joblib.load('randomForestClassifierModel.joblib')
    model_tree = joblib.load('treeClassifierModel.joblib')

    x_test = pd.read_csv('./x_test.csv')
    y_test = pd.read_csv('./y_test.csv')

    accuracy_forest, conf_matrix_forest, class_report_forest = evaluate_model(model_forest, x_test, y_test)
    accuracy_tree, conf_matrix_tree, class_report_tree = evaluate_model(model_tree, x_test, y_test)

    print(f"Relatório e matriz de confusão da Random Forest Model: ")
    print("\n Accuracy: ")
    print(accuracy_forest)
    print("\n Matriz de Confusão")
    print(conf_matrix_forest)
    print("\n Relatório de Classificação")
    print(class_report_forest)

    print("------------------------------------------------------------------------------")

    print(f"\n Relatório e matriz de confusão da Decision Tree Model: ")
    print("\n Accuracy")
    print(accuracy_tree)
    print("\n Matriz de Confusão")
    print(conf_matrix_tree)
    print("\n Relatório de Classificação")
    print(class_report_tree)

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    return accuracy, conf_matrix, class_report

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    main()