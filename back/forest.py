import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
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

    # View table
    x = pd.DataFrame(x)
    x.sample(5)

    # View the array
    lbl = LabelEncoder()
    lbl.fit_transform(y)

    # Split the data into training and testing sets
    train_model(x,y)

def train_model(x,y):
    smote = SMOTE(sampling_strategy='auto', random_state=None)
    x_resampled, y_resampled = smote.fit_resample(x,y)
    # print("Original Dataset Shape:")
    # print(x.shape)
    # print("Resampled Dataset:")
    # print(x_resampled.shape)
    x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.3, random_state=None, shuffle=True, stratify=None)
    randomForestClassifier(x_train, x_test, y_train, y_test)


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
    

# def plot_confusion_matrix(y_test, y_pred, classes):
#     cm = confusion_matrix(y_test, y_pred)
#     plt.figure(figsize=(10, 7))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#     plt.show()


def load_model(filename):
    loaded_model = load(filename)
    return loaded_model

if __name__ == "__main__":
    main()
