import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from joblib import dump,load
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

def main():
    data = pd.read_csv('./frutas.csv', sep=',', header=0)
    data["classe"].unique()
    x = data.drop(['classe'], axis =1)
    y = data['classe']
    print(x)

    # View table
    x = pd.DataFrame(x)
    x.sample(5)
    print(x)

    # View the array
    lbl = LabelEncoder()
    lbl.fit_transform(y)

    # View the array
    view_table(x,y)

    # Split the data into training and testing sets
    train_model(x,y)

def view_table(x, y):
    print(x)
    print(y)

def train_model(x,y):
    smote = SMOTE(sampling_strategy='auto', random_state=None)
    x_resampled, y_resampled = smote.fit_resample(x,y)
    print("Original Dataset Shape:")
    print(x.shape)
    print("Resampled Dataset:")
    print(x_resampled.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=None, shuffle=True, stratify=None)
    treeClassifier(x_train, x_test, y_train, y_test)
    randomForestClassifier(x_train, x_test, y_train, y_test)


def treeClassifier(x_train, x_test, y_train, y_test) :   
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 8, 9, 10, 11, 12, 15, 20],
        'min_samples_split': [2, 3, 4, 5, 10, 15],
        'min_samples_leaf': [2, 3, 4, 5, 6, 10]
    }

    grid_search_dt = RandomizedSearchCV(estimator=DecisionTreeClassifier(), param_distributions=param_grid_dt, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search_dt.fit(x_train, y_train)
    
    print("Grid for the Decision Tree")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search_dt.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search_dt.best_score_)
    print(" ")

    model =  DecisionTreeClassifier(**grid_search_dt.best_params_)

    model.fit(x_train , y_train)

    # script DE COMPARAÇÃO DO MODELO
    np.savetxt('tree_y_test.csv', y_test, delimiter=',', fmt='%s')
    np.savetxt('tree_x_test.csv', x_test, delimiter=',', fmt='%s')

    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    # plot_confusion_matrix(y_test, y_pred, classes=['BERHI', 'DEGLET', 'DOKOL', 'IRAQI', 'ROTANA', 'SAFAVI', 'SOGAY'])
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))

    # errors = np.where(y_test != y_pred)[0]
    # print("Exemplos de erros:")
    # for i in errors[:5]:  # Exibindo os primeiros 5 erros
    #     print(f"Index: {i}, Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}, Features: {x_test.iloc[i]}")
    print("--------------------------------------------------------")

    importances = model.feature_importances_
    features = x_train.columns
    feature_importance = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)
    print(feature_importance)
    
    # save the model in file
    dump(model, 'treeClassifierModel.joblib')

def randomForestClassifier(x_train, x_test, y_train, y_test) :

    param_grid = {
        'bootstrap' : [True, False],
        'n_estimators': [95,100, 105],
        'max_depth': [None, 8, 10, 15],
        'min_samples_split': [2, 4, 3, 5, 6, 7],
        'min_samples_leaf': [2, 3, 4]
    }

    grid_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_grid, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(x_train,y_train)
    print("Grid for the RandomForest")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search.best_score_)
    print(" ")

    model = RandomForestClassifier(**grid_search.best_params_)
    model.fit(x_train , y_train)

    cv_scores = cross_val_score(model, x_train, y_train, cv=5)
    print(f"Cross-Validation Scores: {cv_scores}")
    print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

    # script DE COMPARAÇÃO DO MODELO
    np.savetxt('forest_y_test.csv', y_test, delimiter=',', fmt='%s')
    np.savetxt('forest_x_test.csv', x_test, delimiter=',', fmt='%s')
    
    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------")
    
    # save the model in file
    dump(model, 'randomForestClassifierModel.joblib')
    

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def load_model(filename):
    loaded_model = load(filename)
    return loaded_model

if __name__ == "__main__":
    main()
