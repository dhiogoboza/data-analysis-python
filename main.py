from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from pandas import DataFrame, read_csv
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

OUPUT_COLUMN_NAME = "Classe"

class_index = 0

def prepare(df):
    global class_index
    index = 0
    for column in df:
        if column == OUPUT_COLUMN_NAME:
            class_index = index
        index += 1

    try:
        os.mkdir("output")
    except FileExistsError:
        pass

def calculate_vif(df_complete, column_to_remove = None):
    """
    https://etav.github.io/python/vif_factor_python.html
    """
    df = df_complete
    if not column_to_remove is None:
        df = df_complete.copy()
        del df[column_to_remove];

    new_columns = df.columns.delete(class_index)
    features = '+'.join(map(str, new_columns))
    y, X = dmatrices(OUPUT_COLUMN_NAME + ' ~' + features, df, return_type='dataframe')

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns
    print(vif.round(1))

def boxplot(df_complete):
    df = df_complete.copy()
    del df[OUPUT_COLUMN_NAME];

    df.boxplot()
    plt.savefig("output/boxplot.png")

def correlation(df):
    print(df.corr(method ='pearson'))

def scale_columns(df, cols_to_scale):
    min_max_scaler = preprocessing.MinMaxScaler()
    for col in cols_to_scale:
        df[col] = pd.DataFrame(min_max_scaler.fit_transform(pd.DataFrame(df[col])),columns=[col])

def normalize(df_complete):
    df = df_complete.copy()
    scaled_df = scale_columns(df, df.columns)
    print(df)

def main():
    df = pd.read_csv("dataset/1.csv")
    prepare(df)

    # print data head
    print("Dataframe:")
    print(df.head())

    print("\nVIF complete dataframe:")
    calculate_vif(df)

    print("\nCorrelation:")
    correlation(df)

    print("\nBoxplot:")
    boxplot(df)

    print("\nNormalized:")
    normalize(df)
    

if __name__ == "__main__":
    main()

























