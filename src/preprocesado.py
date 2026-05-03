import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_bar_frecuencias(serie, titulo, xlabel="", ylabel="Frecuencia", top_n=None, rot=45):
    conteo = serie.value_counts(dropna=False)
    if top_n is not None:
        conteo = conteo.head(top_n)
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x=conteo.index.astype(str), y=conteo.values)
    ax.set_title(titulo)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=rot, ha="right")
    plt.tight_layout()
    plt.show()

def plot_hist_box(df, col, bins=30, titulo=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[col], kde=True, bins=bins, ax=axes[0])
    axes[0].set_title(f"Distribución de {col}" if titulo is None else titulo)
    axes[0].set_xlabel(col)
    sns.boxplot(x=df[col], ax=axes[1])
    axes[1].set_title(f"Boxplot de {col}")
    axes[1].set_xlabel(col)
    plt.tight_layout()
    plt.show()

def plot_confusion(y_true, y_pred, etiquetas, titulo):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=etiquetas, yticklabels=etiquetas
    )
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

def plot_real_vs_pred(y_true, y_pred, nombre_modelo):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=15)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='tomato', linestyle='--')
    ax.set_xlabel('Valor real')
    ax.set_ylabel('Predicción')
    ax.set_title(f'Real vs predicción — {nombre_modelo}')
    plt.tight_layout()
    plt.show()

def plot_residuos(y_true, y_pred, nombre_modelo):
    residuos = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(residuos, bins=30, edgecolor='white')
    axes[0].axvline(0, color='tomato', linestyle='--')
    axes[0].set_title('Distribución de residuos')
    axes[0].set_xlabel('Residuo')
    axes[0].set_ylabel('Frecuencia')

    axes[1].scatter(y_pred, residuos, alpha=0.4, s=15)
    axes[1].axhline(0, color='tomato', linestyle='--')
    axes[1].set_title('Residuos vs predicción')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Residuo')

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[2].scatter(y_true, y_pred, alpha=0.4, s=15)
    axes[2].plot([min_val, max_val], [min_val, max_val], color='tomato', linestyle='--')
    axes[2].set_title('Valor real vs predicción')
    axes[2].set_xlabel('Valor real')
    axes[2].set_ylabel('Predicción')

    plt.suptitle(f'Análisis de residuos — {nombre_modelo}', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_varianza_pca(var_exp_total):
    var_acumulada = np.cumsum(var_exp_total)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(var_exp_total)+1), var_exp_total * 100,
                color=sns.color_palette('muted', len(var_exp_total)))
    axes[0].set_xlabel('Componente principal')
    axes[0].set_ylabel('Varianza explicada (%)')
    axes[0].set_title('Varianza explicada por componente')

    axes[1].plot(range(1, len(var_acumulada)+1), var_acumulada * 100,
                 marker='o', linewidth=2)
    axes[1].axhline(80, color='tomato', linestyle='--', alpha=0.7, label='80%')
    axes[1].axhline(90, color='orange', linestyle='--', alpha=0.7, label='90%')
    axes[1].set_xlabel('Número de componentes')
    axes[1].set_ylabel('Varianza acumulada (%)')
    axes[1].set_title('Varianza explicada acumulada')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    for i, (v, va) in enumerate(zip(var_exp_total * 100, var_acumulada * 100)):
        print(f'  PC{i+1}: {v:.1f}%  (acumulada: {va:.1f}%)')


def plot_clusters_pca(X_pca2, var_exp2, labels, centroides, df_objetivo, k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clusters K-Means
    palette = sns.color_palette('muted', k)
    for i in range(k):
        mask = labels == i
        axes[0].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                        c=[palette[i]], label=f'Cluster {i}', alpha=0.5, s=15)
    centroides_pca = centroides @ # necesita los componentes — ver nota abajo
    axes[0].scatter(centroides_pca[:, 0], centroides_pca[:, 1],
                    c='black', marker='X', s=100, zorder=5, label='Centroides')
    axes[0].set_xlabel(f'PC1 ({var_exp2[0]*100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({var_exp2[1]*100:.1f}%)')
    axes[0].set_title(f'Clusters K-Means (k={k})')
    axes[0].legend()

    # Clases reales
    clases = df_objetivo.unique()
    colores = dict(zip(clases, sns.color_palette('Set2', len(clases))))
    for clase in clases:
        mask = df_objetivo == clase
        axes[1].scatter(X_pca2[mask, 0], X_pca2[mask, 1],
                        c=[colores[clase]], label=clase, alpha=0.4, s=15)
    axes[1].set_xlabel(f'PC1 ({var_exp2[0]*100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({var_exp2[1]*100:.1f}%)')
    axes[1].set_title('Clases reales (objetivo)')
    axes[1].legend()

    plt.suptitle('Comparación: clusters K-Means vs clases reales', fontsize=13)
    plt.tight_layout()
    plt.show()