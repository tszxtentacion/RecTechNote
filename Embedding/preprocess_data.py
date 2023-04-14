import os
import urllib.request
import pandas as pd
import numpy as np

data_url = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"


def download_and_extract_data():
    if not os.path.exists("dac.tar.gz"):
        urllib.request.urlretrieve(data_url, "dac.tar.gz")
    if not os.path.exists("train.txt"):
        os.system("tar -xvzf dac.tar.gz")


download_and_extract_data()


def preprocess_data():
    df_train = pd.read_csv("train.txt", header=None, sep="\t")
    df_train.columns = ["clicked"] + [f"I{i}" for i in range(1, 14)] + [f"C{i}" for i in range(1, 27)]
    categorical_cols = [f"C{i}" for i in range(1, 27)]
    continuous_cosl = [f"I{i}" for i in range(1, 14)]
    for col in categorical_cols:
        df_train[col] = df_train[col].fillna("NA")
    for col in continuous_cosl:
        df_train[col] = df_train[col].fillna(df_train[col].mean())
    return df_train, categorical_cols, continuous_cosl


df_train, categorical_cols, continuous_cols = preprocess_data()
