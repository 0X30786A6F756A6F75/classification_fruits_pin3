import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from joblib import dump,load

def main():
    data = pd.read_csv('./frutas.csv', sep=',', header=0)
    data["classe"].unique()
    x = data.drop(['classe'], axis =1)
    y = data['classe']

    x = minmax_scale(x)
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
    rus = RandomUnderSampler(sampling_strategy='majority', replacement=True, random_state=None)
    x_resampled, y_resampled = rus.fit_resample(x,y)
    print("Original Dataset Shape:")
    print(x.shape)
    print("Resampled Dataset:")
    print(x_resampled.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=None, shuffle=True, stratify=None)
    treeClassifier(x_train, x_test, y_train, y_test)
    randomForestClassifier(x_train, x_test, y_train, y_test)


def treeClassifier(x_train, x_test, y_train, y_test) :   
    model =  DecisionTreeClassifier(max_depth=None, criterion='gini', splitter='best')
    param_grid_dt = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid_dt, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search_dt.fit(x_train, y_train)
    
    print("Grid for the Decision Tree")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search_dt.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search_dt.best_score_)
    print(" ")

    model.fit(x_train , y_train)

    # save the model in file
    dump(model, 'treeClassifierModel.joblib')
    
    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------")

def randomForestClassifier(x_train, x_test, y_train, y_test) :
    model = RandomForestClassifier(n_estimators=100, max_depth=None, criterion='gini', bootstrap=True)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'bootstrap' : [True, False],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=1)
    grid_search.fit(x_train,y_train)
    print("Grid for the RandomForest")
    print(" ")
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    print(" ")
    print("Performance on the Validation Set:")
    print(grid_search.best_score_)
    print(" ")
    model.fit(x_train , y_train)

    # save the model in file
    dump(model, 'randomForestClassifierModel.joblib')
    
    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------")
    
def load_model(filename):
    loaded_model = load(filename)
    return loaded_model

if __name__ == "__main__":
    main()
