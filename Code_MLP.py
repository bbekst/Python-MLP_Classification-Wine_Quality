#import libraries
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from os import system

#clear screen
system("clear")

#load data
df = pd.read_csv('wine.csv')

#define features & labels converting labels to binary numbers
X = df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol']]
Y = df['quality'].replace({'good': 1, 'bad': 0})

#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#define function to fit model, predict, & metrics
def mlp(models, X_true, Y_true, combinations, dataset_name):

    print(f'{dataset_name} Dataset:')
    for model, combination in zip(models, combinations):

        model.fit(X_train, Y_train)
        Y_proba = model.predict_proba(X_true)[:, 1]
        Y_pred = Y_proba.round()

        print(f"{combination}")
        print("Accuracy:", round(metrics.accuracy_score(Y_true, Y_pred), 4))
        print("Sensitivity:", round(metrics.recall_score(Y_true, Y_pred, pos_label=1), 4))
        print("Specificity:", round(metrics.recall_score(Y_true, Y_pred, pos_label=0), 4))
        print("F1 Score:", round(metrics.f1_score(Y_true, Y_pred, pos_label=1), 4))
        print("Log Loss:", round(metrics.log_loss(Y_true, Y_proba), 4))

        fpr, tpr, _ = metrics.roc_curve(Y_true, Y_proba)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{combination} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {dataset_name} Dataset')
    plt.legend(loc="lower right")
    plt.show()
    print(" ")

#define models with three different combinations
models = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), activation='relu', max_iter=5000, random_state=1),
          MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3, 2), activation='relu', max_iter=5000, random_state=1),
          MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(5, 2), activation='tanh', max_iter=5000, random_state=1)]

#combination names
combinations = ['Combination 1', 'Combination 2', 'Combination 3']

#call function for each dataset
mlp(models, X_train, Y_train, combinations, 'Training')
mlp(models, X_test, Y_test, combinations, 'Test')