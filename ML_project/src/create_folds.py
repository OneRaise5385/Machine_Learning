import numpy as np
import pandas as pd
from sklearn import model_selection

if __name__ == '__main__':
    # 导入数据
    df_x = np.loadtxt('../input/mnist_x')
    df_y = np.loadtxt('../input/mnist_y')
    df = pd.DataFrame(df_x)
    df['label'] = df_y
    df['kfold'] = -1

    # 随机打乱数据
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=5)
    for fold, (trn_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    # 保存新的数据集
    df.to_csv('../input/mnist.csv', index=False)