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

def main():
    data = pd.read_csv('./frutas.csv', sep=',', header=0)
    data["classe"].unique()
    X = data.drop(['classe'], axis =1)
    Y = data['classe']
    type(X)

    X = minmax_scale(X)
    print(X)

    # View table
    X = pd.DataFrame(X)
    X.sample(5)

    # View the array
    lbl = LabelEncoder()
    lbl.fit_transform(Y)

    # View the array
    view_table(X,Y)

    # Split the data into training and testing sets
    train_model(X,Y)

def view_table(X, Y):
    print(X)
    print(Y)

def train_model(X,Y):
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=42, shuffle=True, stratify=None)

    # Train the model
    models = [
        DecisionTreeClassifier(),  
        RandomForestClassifier()  
    ]
    for model in models : 
        model.fit (X_train , y_train )
        
        y_pred = model.predict(X_test)
        
        print(f"Algoritmo: {model.__class__.__name__}")
        print("Matriz de Confusão")
        print(confusion_matrix(y_test, y_pred))
        print("\n Relatório de Classificação")
        print(classification_report(y_test, y_pred))
        print("--------------------------------------------------------")

if __name__ == "__main__":
    main()
