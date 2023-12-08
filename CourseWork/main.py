# 导入模块
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import sklearn.tree as tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss, mean_squared_log_error
import sys, os, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# 数据导入
train_file = os.listdir('input/train/')
test_file = os.listdir('input/test/')
train = pd.DataFrame()
test = pd.DataFrame()
# 读取训练数据
for file in train_file:
    tmp = pd.read_csv('input/train/'+file)
    tmp['file'] = file
    train = pd.concat([train, tmp], axis=0, ignore_index=True)
# 读取测试数据
for file in test_file:
    tmp = pd.read_csv('input/test/'+file)
    tmp['file'] = file
    test = pd.concat([test, tmp], axis=0, ignore_index=True)

# 查看训练集信息
train.info()
train.head()
train.tail()
train.describe()

# 描述性分析
# 1
cols = ['n_bid1','n_bid2','n_ask1','n_ask2']
tmp_df = train[train['file']=='snapshot_sym1_date0_am.csv'].reset_index(drop=True)[-2000:]
tmp_df = tmp_df.reset_index(drop=True).reset_index()
for num, col in enumerate(cols):
    plt.figure(figsize=(15,5))
    plt.subplot(4,1,num+1)
    plt.plot(tmp_df['index'],tmp_df[col], color='red')
    plt.title(col)
plt.show()

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)
plt.legend(fontsize=12)

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)

plt.plot(tmp_df['index'],tmp_df['n_midprice'],label="n_midprice",lw=3)
plt.legend(fontsize=12)

# 2
cols = ['n_bid3','n_bid4','n_ask3','n_ask4']
tmp_df = train[train['file']=='snapshot_sym1_date0_am.csv'].reset_index(drop=True)[-2000:]
tmp_df = tmp_df.reset_index(drop=True).reset_index()
for num, col in enumerate(cols):
    plt.figure(figsize=(15,5))
    plt.subplot(4,1,num+1)
    plt.plot(tmp_df['index'],tmp_df[col], color='red')
    plt.title(col)
plt.show()

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)
plt.legend(fontsize=12)

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)

plt.plot(tmp_df['index'],tmp_df['n_midprice'],label="n_midprice",lw=3)
plt.legend(fontsize=12)

# 3
cols = ['n_bid5','n_bid4','n_ask5','n_ask4']
tmp_df = train[train['file']=='snapshot_sym1_date0_am.csv'].reset_index(drop=True)[-2000:]
tmp_df = tmp_df.reset_index(drop=True).reset_index()
for num, col in enumerate(cols):
    plt.figure(figsize=(15,5))
    plt.subplot(4,1,num+1)
    plt.plot(tmp_df['index'],tmp_df[col], color='red')
    plt.title(col)
plt.show()

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)
plt.legend(fontsize=12)

plt.figure(figsize=(15,5))
for num, col in enumerate(cols):
    plt.plot(tmp_df['index'],tmp_df[col],label=col)

plt.plot(tmp_df['index'],tmp_df['n_midprice'],label="n_midprice",lw=3)
plt.legend(fontsize=12)

# 4
cols = ['amount_delta']
tmp_df = train[train['file']=='snapshot_sym1_date0_am.csv'].reset_index(drop=True)[1:500]
tmp_df = tmp_df.reset_index(drop=True).reset_index()
for num, col in enumerate(cols):
    plt.figure(figsize=(15,50))
    plt.subplot(4,1,num+1)
    plt.plot(tmp_df['index'],tmp_df[col])
    plt.title(col)
plt.show()

# 5
train['wap1'] = (train['n_bid1']*train['n_bsize1'] + train['n_ask1']*train['n_asize1'])/(train['n_bsize1'] + train['n_asize1'])
test['wap1'] = (test['n_bid1']*test['n_bsize1'] + test['n_ask1']*test['n_asize1'])/(test['n_bsize1'] + test['n_asize1'])

tmp_df = train[train['file']=='snapshot_sym1_date0_am.csv'].reset_index(drop=True)[-2000:]
tmp_df = tmp_df.reset_index(drop=True).reset_index()
plt.figure(figsize=(20,10))
plt.plot(tmp_df['index'], tmp_df['wap1'], color='red')
plt.ylabel('Price Volatility')

# 时间相关特征
train['hour'] = train['time'].apply(lambda x:int(x.split(':')[0]))
test['hour'] = test['time'].apply(lambda x:int(x.split(':')[0]))

train['minute'] = train['time'].apply(lambda x:int(x.split(':')[1]))
test['minute'] = test['time'].apply(lambda x:int(x.split(':')[1]))

# 排序
train = train.sort_values(['file','time'])
test = test.sort_values(['file','time'])

# 当前时间特征
# 构建买一卖一和买二卖二相关特征
train['wap1'] = (train['n_bid1']*train['n_bsize1'] + train['n_ask1']*train['n_asize1'])/(train['n_bsize1'] + train['n_asize1'])
test['wap1'] = (test['n_bid1']*test['n_bsize1'] + test['n_ask1']*test['n_asize1'])/(test['n_bsize1'] + test['n_asize1'])

train['wap2'] = (train['n_bid2']*train['n_bsize2'] + train['n_ask2']*train['n_asize2'])/(train['n_bsize2'] + train['n_asize2'])
test['wap2'] = (test['n_bid2']*test['n_bsize2'] + test['n_ask2']*test['n_asize2'])/(test['n_bsize2'] + test['n_asize2'])

train['wap_balance'] = abs(train['wap1'] - train['wap2'])
train['price_spread'] = (train['n_ask1'] - train['n_bid1']) / ((train['n_ask1'] + train['n_bid1'])/2)
train['bid_spread'] = train['n_bid1'] - train['n_bid2']
train['ask_spread'] = train['n_ask1'] - train['n_ask2']
train['total_volume'] = (train['n_asize1'] + train['n_asize2']) + (train['n_bsize1'] + train['n_bsize2'])
train['volume_imbalance'] = abs((train['n_asize1'] + train['n_asize2']) - (train['n_bsize1'] + train['n_bsize2']))

test['wap_balance'] = abs(test['wap1'] - test['wap2'])
test['price_spread'] = (test['n_ask1'] - test['n_bid1']) / ((test['n_ask1'] + test['n_bid1'])/2)
test['bid_spread'] = test['n_bid1'] - test['n_bid2']
test['ask_spread'] = test['n_ask1'] - test['n_ask2']
test['total_volume'] = (test['n_asize1'] + test['n_asize2']) + (test['n_bsize1'] + test['n_bsize2'])
test['volume_imbalance'] = abs((test['n_asize1'] + test['n_asize2']) - (test['n_bsize1'] + test['n_bsize2']))

# 历史平移
# 获取历史信息
for val in ['wap1','wap2','wap_balance','price_spread','bid_spread','ask_spread','total_volume','volume_imbalance']:
    for loc in [1,5,10,20,40,60]:
        train[f'file_{val}_shift{loc}'] = train.groupby(['file'])[val].shift(loc)
        test[f'file_{val}_shift{loc}'] = test.groupby(['file'])[val].shift(loc)
    
# 差分特征
# 获取与历史数据的增长关系
for val in ['wap1','wap2','wap_balance','price_spread','bid_spread','ask_spread','total_volume','volume_imbalance']:
    for loc in [1,5,10,20,40,60]:
        train[f'file_{val}_diff{loc}'] = train.groupby(['file'])[val].diff(loc)
        test[f'file_{val}_diff{loc}'] = test.groupby(['file'])[val].diff(loc)
    
# 获取历史信息分布变化信息
for val in ['wap1','wap2','wap_balance','price_spread','bid_spread','ask_spread','total_volume','volume_imbalance']:
    train[f'file_{val}_win7_mean'] = train.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).mean())
    train[f'file_{val}_win7_std'] = train.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).std())
    
    test[f'file_{val}_win7_mean'] = test.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).mean())
    test[f'file_{val}_win7_std'] = test.groupby(['file'])[val].transform(lambda x: x.rolling(window=7, min_periods=3).std())

# 训练函数
def cv_model(clf, train_x, train_y, test_x, test_y, clf_name, seed = 2023):
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    test_predict = np.zeros([test_x.shape[0], 3])
    
    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('{}* is training'.format(str(i+1)))
        trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], train_y[valid_index]

        if clf_name == "xgb":
            # xgboost
            xgb_params = {
              'booster': 'gbtree', 
              'objective': 'multi:softprob',
              'num_class':3,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.1,
              'tree_method': 'hist',
              'seed': 2023,
              'nthread': 16,
              }
            train_matrix = clf.DMatrix(trn_x , label=trn_y)
            valid_matrix = clf.DMatrix(val_x , label=val_y)
            test_matrix = clf.DMatrix(test_x)
            
            watchlist = [(train_matrix, 'train'),(valid_matrix, 'eval')]
            
            model = clf.train(xgb_params, train_matrix, num_boost_round=200, evals=watchlist)
            val_pred  = model.predict(valid_matrix)
            test_pred = model.predict(test_matrix)
            
        if clf_name == "c45":
            # c45决策树
            model = clf(criterion='entropy',splitter='best',max_depth=5)
            model.fit(trn_x, trn_y)
            
            val_pred  = model.predict_proba(val_x)
            test_pred = model.predict_proba(test_x)
        
        if clf_name == "cart":
            # cart决策树
            model = clf(criterion='gini',splitter='best',max_depth=5)
            model.fit(trn_x, trn_y)
            
            val_pred  = model.predict_proba(val_x)
            test_pred = model.predict_proba(test_x)

        test_predict += test_pred / kf.n_splits

    test_label = np.argmax(test_predict, axis=1)
    F1_score = f1_score(test_y, test_label, average='micro')
    print('F1_score:',F1_score)
    return F1_score

# 处理train_x和test_x中的NaN值
train = train.fillna(0)
test = test.fillna(0)

# 处理train_x和test_x中的Inf值
train = train.replace([np.inf, -np.inf], 0)
test = test.replace([np.inf, -np.inf], 0)

# 把作预测的数据集的特征提取出来
cols_x = [f for f in test.columns if f not in 
          ['uuid','time','file','label_5','label_10','label_20','label_40','label_60']]

# 开始训练
cols = [f for f in test.columns if f not in ['uuid','time','file']]
for label in ['label_5','label_10','label_20','label_40','label_60']:
# for label in ['label_5']:
    print(f'==== {label} ====')
    # 选择c4.5模型
    c45_test = cv_model(tree.DecisionTreeClassifier,
                        train[cols_x], train[label], test[cols_x], test[label], 'c45')
    # 选择cart模型
    cart_test = cv_model(tree.DecisionTreeClassifier,
                         train[cols_x], train[label], test[cols_x], test[label], 'cart')
    
    # 选择随机森林算法
    cart_test = cv_model(RandomForestClassifier,
                         train[cols_x], train[label], test[cols_x], test[label], 'rf')
    
    # 选择xgboost模型
    xgb_test = cv_model(xgb, train[cols], train[label], test[cols], test[label], 'xgb')

# 绘制得分图
f1 = pd.read_csv('input/f1_score.csv')
plt.figure(figsize=(24,13))
plt.subplot(221)
plt.plot(f1['label'],f1['c45'],color='blue', ls='--', label='c45 score')
plt.plot(f1['label'],f1['cart'],color='red', ls='-.', label='cart score')
plt.plot(f1['label'],f1['random forest'],color='purple', ls=':', label='random forest')
plt.plot(f1['label'],f1['xgboost'],color='green', label='xgboost score')
plt.xlabel('Labels')
plt.ylabel('F1 score')
plt.legend()