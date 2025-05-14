import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

def read_data(filepath):
    colunas = ['id', 'pSist', 'pDiast', 'qPA', 'pulso', 'resp', 'gravidade', 'classe']
    df = pd.read_csv(filepath, header=None, names=colunas, decimal=".")

    return df[['qPA', 'pulso', 'resp', 'classe']]


def cross_validate_id3(X, y, folds=5):
    kf = KFold(n_splits=folds, shuffle=True)
    fold = 1
    accuracies = []

    for train_index, test_index in kf.split(X):
        print(f"\n--- Fold {fold} ---")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = DecisionTreeClassifier(criterion="entropy")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4])
        report = classification_report(y_test, y_pred, labels=[1, 2, 3, 4], zero_division=0)

        print(f"Acurácia: {acc:.2f}")
        print("Matriz de Confusão:")
        print(cm)
        print("Relatório de Classificação:")
        print(report)

        accuracies.append(acc)
        fold += 1

    print("\n--- Resultado Final ---")
    print(f"Acurácia Média: {np.mean(accuracies):.2f}")


def normalize(df):
    scaler = MinMaxScaler()
    features = df.drop(columns=["classe"])
    features_scaled = scaler.fit_transform(features)
    return pd.DataFrame(features_scaled, columns=features.columns), df["classe"].values


def algorithm_id3():
    data = read_data("./assets/treino_sinais_vitais_com_label.txt")
    X_df, y = normalize(data)
    X = X_df.values
    cross_validate_id3(X, y, folds=5)


algorithm_id3()

