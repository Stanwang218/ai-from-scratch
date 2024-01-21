import pandas as pd
import numpy as np
import sys
import os

os.chdir(sys.path[0])
        

def ShannonEntropy(data:pd.Series):
    l = len(data)
    value_cnt = data.value_counts().to_dict()
    # print(value_cnt)
    entropy_calculator = lambda d: sum(-(p / l) * np.log2(p / l) for p in d.values())
    return entropy_calculator(value_cnt)

class node():
    def __init__(self, attr = None, value = None):
        self.child = dict()
        self.attr = attr
        self.value = value # type
        self.leaf = False
    
class DecisionTree():
    def __init__(self, dataPath) -> None:
        self.data = pd.read_csv(dataPath)
        self.colName = self.data.columns
        self.targetCol = self.colName[-1]
        self.selectedFeature = set(self.colName[:-1])
        self.root = node()
        
    def computeConditionalEntropy(self, data:pd.DataFrame,attr):
        value_cnt = data[attr].value_counts().to_dict()
        l = len(data)
        condEntropy = 0
        entropy_list = []
        for key in value_cnt.keys():
            selected_data = data[data[attr] == key][self.targetCol]
            temp_entropy = ShannonEntropy(selected_data)
            condEntropy += temp_entropy * len(selected_data) / l
            entropy_list.append(temp_entropy)
        # print(entropy_list)
        # print(condEntropy)
        return condEntropy
    
    def generateBranch(self, data:pd.DataFrame, featureSet, root:node):
        data_type = data[self.targetCol].unique()
        if len(data_type) == 1:
            root.leaf = True
            root.value = data_type[0]
            return
        highest_value = 0
        seleted_feature = None
        for feature in featureSet:
            temp_value = ShannonEntropy(data[self.targetCol]) - self.computeConditionalEntropy(data, feature)
            # print(f"{feature} : {self.computeConditionalEntropy(self.data, feature)}")
            if seleted_feature is None:
                highest_value = temp_value
                seleted_feature = feature
            else:
                if highest_value < temp_value:
                    highest_value = temp_value
                    seleted_feature = feature
        value_cnt = data[seleted_feature].value_counts().to_dict()
        root.attr = seleted_feature
        for key in value_cnt.keys():
            root.child[key] = node()
            temp_data = data[data[seleted_feature] == key]
            self.generateBranch(temp_data, featureSet - {seleted_feature} ,root.child[key])
            
def showDecisionTree(root:node, path: list):
    for key in root.child:
        path.extend([root.attr, key])
        if not root.child[key].leaf:
            showDecisionTree(root.child[key], path)
        else:
            print(f"{path} -> {root.child[key].value}")
        path.pop()
        path.pop()
            
            
if __name__ == "__main__":
    # path = './melon.csv'
    path = './id3_data.csv'
    tree = DecisionTree(path)
    data = tree.data
    col = tree.colName
    tree.generateBranch(tree.data, tree.selectedFeature, tree.root)
    print(data.drop_duplicates())
    showDecisionTree(tree.root, [])

    # # Problem 2 : base on the attribute of sense of taste
    # attr = data[col[2]].value_counts().to_dict()
    # for item in attr:
    #     print(data[data[col[2]] == item])

    # # Problem 3 : compute the conditional entropy
    # tree.computeConditionalEntropy(tree.data, col[2])