import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, StratifiedKFold


def load_data(path="./data"):
    print("Loading data...")
    train_csv = os.path.join(path, "train.csv")
    test_csv = os.path.join(path, "test.csv")
    train_raw_data_frame = pd.read_csv(train_csv)
    test_raw_data_frame = pd.read_csv(test_csv)
    return train_raw_data_frame, test_raw_data_frame


def feature_eng(data_raw):
    print("Evaluating and choosing features...")
    # 选择特征
    corr = data_raw.corr()
    print("Correlation of each feature and the result:\n", corr["Survived"].sort_values(ascending=False))
    data_features = data_raw[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()  # 舍弃了Cabin(缺省值太多)
    # 属性组合
    data_features["Relative"] = pd.DataFrame(
        {"Relative": [1 if i == 1 or j == 1 else 0 for i, j in zip(data_features["SibSp"], data_features["Parch"])]})
    print("Final features set are:\n", data_features.columns)
    # 提取标签
    print("Extracting labels...")
    data_labels = data_raw["Survived"].copy()
    # 分割数字和类别特征
    print("Processing feature types...")
    data_num = data_features[["Age", "Fare", "Pclass", "Relative"]]
    data_str = data_features[["Sex", "Embarked"]]
    # 处理数字缺省值
    num_imputer = SimpleImputer(strategy="median")
    data_num_numpy = num_imputer.fit_transform(data_num)
    data_num = pd.DataFrame(data_num_numpy, columns=data_num.columns)
    # 处理文本分类缺省值及数字转化
    str_imputer = SimpleImputer(strategy="most_frequent")
    data_str_numpy = str_imputer.fit_transform(data_str)
    data_str = pd.DataFrame(data_str_numpy, columns=data_str.columns)
    encoder = LabelBinarizer()
    data_sex_1hot = encoder.fit_transform(data_str["Sex"])
    data_embarked_1hot = encoder.fit_transform(data_str["Embarked"])
    data_cat_numpy = np.c_[data_sex_1hot, data_embarked_1hot]

    data_set_numpy = np.c_[data_num_numpy, data_cat_numpy]
    data_label_numpy = np.array(data_labels)
    print("Features ready")
    return data_set_numpy, data_label_numpy

def model_training(train_set, train_labels, algorithm="LogisticReg", cross_validation=True):
    print("Model algorithm used is:", algorithm)
    if algorithm == "LogisticReg":
        if cross_validation:
            model = LogisticRegressionCV(cv=StratifiedKFold(n_splits=3, shuffle=False, random_state=None), max_iter=1000)
        else:
            model = LogisticRegression()
    if algorithm == "SGD":
        model = SGDClassifier(max_iter=1000, early_stopping=True)
    print("Model training started...")
    model.fit(train_set, train_labels)
    print("Model training completed!")
    return model

def cross_validation(model, data_set, labels):
    print("Using Cross Validation to measure model performance...")
    scores = cross_val_score(model, data_set, labels, scoring="neg_mean_squared_error", cv=3)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores

def model_save(model, model_name):
    file = "model/" + model_name + "_" + time.strftime("%Y%m%d", time.localtime()) + ".model"
    output = open(file, 'wb')
    pickle.dump(model, output)
    output.close()
    print("Model",file,"Saved!")

def model_reload(file):
    model_file = open(file, 'rb')
    model = pickle.load(model_file)
    print("Model reloaded!")
    return model

if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    # 读取数据
    train_raw, test_raw = load_data()

    # 数据处理及特征抽取
    train_set, train_labels = feature_eng(train_raw)


    # 训练模型
    model_name = "SGD"
    model = model_training(train_set, train_labels, algorithm=model_name, cross_validation=True)
    # 存储模型
    model_save(model, model_name)



    # 读取模型
    model_file = "model/SGD_20191218.model"
    model = model_reload(model_file)
    predictions = model.predict(train_set)
    lin_rmse = np.sqrt(mean_squared_error(predictions, train_labels))
    print("RMSE:", lin_rmse)
    print("Model score:", model.score(train_set, train_labels))

    cross_val_scores = cross_validation(model, train_set, train_labels)
    print("Scores:", cross_val_scores)
    print("Mean:", cross_val_scores.mean())
    print("Standard Deviation:", cross_val_scores.std())
    print(model.n_iter_)
