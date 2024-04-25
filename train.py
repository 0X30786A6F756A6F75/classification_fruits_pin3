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

    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=42, shuffle=True, stratify=None)
    treeClassifier(x_train, x_test, y_train, y_test)
    randomForestClassifier(x_train, x_test, y_train, y_test)


def treeClassifier(x_train, x_test, y_train, y_test) :
    model =  DecisionTreeClassifier(random_state=42, max_depth=None, criterion='gini', splitter='best')
    model.fit(x_train , y_train)
    
    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------")

def randomForestClassifier(x_train, x_test, y_train, y_test) :
    model =  RandomForestClassifier(n_estimators=100, random_state=42, max_depth=None, criterion='gini')
    model.fit(x_train , y_train)
    
    y_pred = model.predict(x_test)
    
    print(f"Algoritmo: {model.__class__.__name__}")
    print("Matriz de Confusão")
    print(confusion_matrix(y_test, y_pred))
    print("\n Relatório de Classificação")
    print(classification_report(y_test, y_pred))
    print("--------------------------------------------------------")
    

if __name__ == "__main__":
    main()
