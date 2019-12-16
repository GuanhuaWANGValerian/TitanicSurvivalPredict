import os
import pandas as pd
import re


def load_data(path="./data"):
    train_csv = os.path.join(path, "train.csv")
    test_csv = os.path.join(path, "test.csv")
    train_raw_data_frame = pd.read_csv(train_csv)
    test_raw_data_frame = pd.read_csv(test_csv)
    return train_raw_data_frame, test_raw_data_frame

def feature_eng(data_raw):
    print(train_raw.info())
    #处理缺省值

# def data_prep(data_raw):


if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    train_raw, test_raw = load_data()
    print(train_raw.head())
    feature_eng(train_raw)
'''
    relative = []
    for i, j in zip(train_raw["Parch"], train_raw["SibSp"]):
        relative.append(1 if i == 1 or j == 1 else 0)
    print(str(relative))
    relative = pd.Series(relative)
    print(relative)
    train_raw["relative"] = relative
    print(train_raw.corr())
    # print(train_raw["relative"])
'''