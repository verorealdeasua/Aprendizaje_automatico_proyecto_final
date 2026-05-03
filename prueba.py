
from sklearn.model_selection import cross_val_score
import numpy as np

def escalar_estandar(x):
    return (x - x.mean()) / x.std()

def one_hot_encode(X, categorical_indices, drop_first=False):

    # pd.get_dummies(df["nacionalidad"])
    # con sklearn: OneHotEncoder(handle_unknown="ignore")

    X_transformed = X.copy()

    for index in sorted(categorical_indices, reverse=True):
        #Extract the categorical column
        categorical_column = X_transformed[:,index]

        #Find the unique categories (works with strings)
        unique_values = np.unique(categorical_column)

        #Create a one-hot encoded matrix (np.array) for the current categorical column
        one_hot = []
        for valor in unique_values:
            columna_binaria = []

            for categoria in categorical_column:
                if categoria == valor:
                    columna_binaria.append(1)
                else:
                    columna_binaria.append(0)

            one_hot.append(columna_binaria)
        one_hot = np.array(one_hot).T #queremos datosxcategorias 

        # Optionally drop the first level of one-hot encoding
        if drop_first:
            one_hot = one_hot[:, 1:]

        #Delete the original categorical column from X_transformed and insert new one-hot encoded columns
        X_sin_columna = np.delete(X_transformed, index, axis=1)
        X_transformed = np.concatenate((X_sin_columna[:,:index], one_hot, X_sin_columna[:,index:]), axis=1)   #queremos añadir la nueva columna en la posicion original de la columna que hemos eliminado

    return X_transformed

def one_hot_encode2(dataset, column):
    unique_values = np.unique(dataset[column])

    #Create a one-hot encoded matrix (np.array) for the current categorical column
    one_hot = []
    for valor in unique_values:
        columna_binaria = []

        for categoria in dataset[column]:
            if categoria == valor:
                columna_binaria.append(1)
            else:
                columna_binaria.append(0)

        one_hot.append(columna_binaria)
    one_hot = np.array(one_hot).T #queremos datosxcategorias 

    #Delete the original categorical column from X_transformed and insert new one-hot encoded columns
    dataset[column] = one_hot
    return dataset


def cross_validation(model, X_train, y_train, valores):
    scores = []
    medias = []

    for v in valores:
        modelo  = model()
        score = cross_val_score(modelo, X_train, y_train)
        scores.append(score)
        medias.append(score.mean())
    return scores

