import os
import config
import joblib
import pandas as pd
from sklearn import metrics
import model_dispatcher
import argparse

def run(fold, model):
    # 读取数据文件
    df = pd.read_csv(config.TRAINING_FILE)

    # 划分训练集与测试集
    df_train = df[df['kfold'] != fold].reset_index(drop=True)
    df_test = df[df['kfold'] == fold].reset_index(drop=True)

    # 训练集特征标签分离
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # 测试集特征标签分离
    x_test = df_test.drop('label', axis=1).values
    y_test = df_test.label.values

    # 决策树模型
    clf = model_dispatcher.models[model]
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    # 计算准确率
    accuracy = metrics.accuracy_score(y_test, preds)
    print(f'Fold={fold}, Accuracy={accuracy}')

    # 保存模型
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, f'df_{fold}.bin'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # fold参数
    parser.add_argument('--fold', type=int)
    # model参数
    parser.add_argument('--model', type=str)
    
    args = parser.parse_args()
    run(fold=args.fold, model=args.model)
