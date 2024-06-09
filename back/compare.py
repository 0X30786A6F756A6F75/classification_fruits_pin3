import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np


def main():
    model_forest  = joblib.load('randomForestClassifierModel.joblib')
    model_tree = joblib.load('treeClassifierModel.joblib')

    x_test_tree = pd.read_csv('./tree_x_test.csv')
    y_test_tree = pd.read_csv('./tree_y_test.csv')

    x_test_forest = pd.read_csv('./forest_x_test.csv')
    y_test_forest = pd.read_csv('./forest_y_test.csv')

    accuracy_forest, conf_matrix_forest, class_report_forest = evaluate_model(model_forest, x_test_forest, y_test_forest)
    accuracy_tree, conf_matrix_tree, class_report_tree = evaluate_model(model_tree, x_test_tree, y_test_tree)

def evaluate_model(model, x_test, y_test):

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(conf_matrix)
    # plot_confusion_matrix(y_test, y_pred, classes=['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY'])
    print("\n Relatório de Classificação")
    print(class_report)

    return accuracy, conf_matrix, class_report


if __name__ == "__main__":
    main()