from sklearn import tree
from sklearn import ensemble
models = {
    # gini决策树
    'decision_tree_gini': tree.DecisionTreeClassifier(criterion='gini'),
    # 熵决策树
    'decision_tree_entropy': tree.DecisionTreeClassifier(criterion='entropy'),
    # 随机森林
    'rf': ensemble.RandomForestClassifier(),
}
