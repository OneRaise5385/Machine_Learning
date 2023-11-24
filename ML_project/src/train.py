import joblib
import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold):
    # 读取数据文件
    df = pd.read_table('../input/')