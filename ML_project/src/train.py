import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # 读取数据文件
    df = pd.read_table('../input/mnist_train_folds.csv')

    # 划分训练集与测试集
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_test = df[df.kfold == fold].reset_index(drop=True)

    # 训练集特征标签分离
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # 测试集特征标签分离
    x_test = df_test.drop('label', axis=1).values
    y_test = df_test.label.values

    # 决策树模型
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    # 计算准确率
    accuracy = metrics.accuracy_score(y_test, preds)
    print(f'Folf={fold}', Accuracy={accuracy})

    # 保存模型
    joblib.dump(clf, f'../models/df_{fold}.bin')

if __name__ == '__main__':
    run(fold=0)
    run(fold=1)
    run(fold=2)
    run(fold=3)
    run(fold=4)