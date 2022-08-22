#imports 

import joblib
from sklearn.tree import _tree
import copy
import pickle
import numpy as np
import json
import argparse
import sys

#Global variables
Nodes_Last_Val = dict() 
Leaves_Decisions = dict()
Leaves_Metada = dict()
Feature_thresholds = dict()


def recurse(model, class_names, node_id=0, parent=None, depth=None):

    """
    Recurse the model's tree and save every leaf node's information (gini, samples, 
    decisions...) starting from the node_id (in most cases will be the root).

    Parameters
    ----------
    model : the decision tree model (sklearn.tree._classes.DecisionTreeClassifier)
    
    class_names : names of the problem classes (strings)
    
    node_id: starting node, in most cases is set to 0 (root) (int)
    
    parent: if node_id has a parent must be provided (int)

    depth: is the depth of the tree, it starts from 0 up to the tree max_depth

    Returns
    ----------
    True

    """

    tree = model.tree_

    if node_id == _tree.TREE_LEAF:
        raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]

    depth = 0
    if depth <= tree.max_depth:

        save_node_info(tree, node_id, parent=parent, class_names=class_names)

        if left_child != _tree.TREE_LEAF:
            recurse(model=model, node_id=left_child, class_names=class_names, parent=node_id, depth=depth + 1)
            recurse(model=model, node_id=right_child, class_names=class_names, parent=node_id, depth=depth + 1)

    return True          

def save_node_info(tree, node_id, class_names, parent=None):

    """
    Save the node's information (impurity, decision, samples, ..) to the global variables.

    Parameters
    ----------
    tree : the decision tree tree (sklearn.tree._tree.Tree)
    
    node_id: starting node (int)
    
    class_names : names of the problem classes (strings)
    
    parent: if node_id has a parent, it must be provided (int)

    Returns
    ----------
    True

    """

    global Nodes_Last_Val
    global Leaves_Decisions
    global Feature_thresholds

    #write decision criteria as range
    if tree.children_left[node_id] != _tree.TREE_LEAF:

        try:
            Feature_thresholds[str(tree.feature[node_id])].append(tree.threshold[node_id])
        except KeyError:
            Feature_thresholds[str(tree.feature[node_id])] = [tree.threshold[node_id]]

        if parent is not None:
            Nodes_Last_Val[str(node_id)] = copy.deepcopy(Nodes_Last_Val[str(parent)])
            try:
                _ = Nodes_Last_Val[str(node_id)][str(tree.feature[node_id])]
            except KeyError:
                Nodes_Last_Val[str(node_id)][str(tree.feature[node_id])] = [None, None]

            #when left node, the parent is constrained by max (<=)
            if node_id == tree.children_left[parent]:
                Nodes_Last_Val[str(node_id)][str(tree.feature[parent])][1] = tree.threshold[parent]
            else:
                Nodes_Last_Val[str(node_id)][str(tree.feature[parent])][0] = tree.threshold[parent]

        #for root node
        else:
            Nodes_Last_Val[str(node_id)] = dict()
            Nodes_Last_Val[str(node_id)][str(tree.feature[node_id])] = [None, None]
    
    #when the node is a leaf
    else:
        Nodes_Last_Val[str(node_id)] = copy.deepcopy(Nodes_Last_Val[str(parent)])

        if node_id == tree.children_left[parent]:
            Nodes_Last_Val[str(node_id)][str(tree.feature[parent])][1] = tree.threshold[parent]
        else:
            Nodes_Last_Val[str(node_id)][str(tree.feature[parent])][0] = tree.threshold[parent]
        
        Leaves_Decisions[str(node_id)] = Nodes_Last_Val[str(node_id)]

        Leaves_Metada[str(node_id)] = dict()
        Leaves_Metada[str(node_id)]['impurity'] = round(tree.impurity[node_id], 3) #need to define what impuritu was used
        Leaves_Metada[str(node_id)]['samples'] = int(tree.n_node_samples[node_id])
        value = tree.value[node_id][0, :]
        Leaves_Metada[str(node_id)]['value'] = list(value)
        Leaves_Metada[str(node_id)]['class'] = class_names[np.argmax(value)]
    
    return True

""" code arguments handler: """

parser = argparse.ArgumentParser(description="Save the model's leaf nodes for later analysis:",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-model_path", help="File location to DT models (serialized: pickle or jiblib)")

parser.add_argument("-to_store_leaves", help="File location where to store the leaf nodes data")
parser.add_argument("-to_store_metadata", help="File location where to store the leaf nodes metadata")
parser.add_argument("-to_store_thresholds", help="File location where to store the features' thresholds/decisions")
parser.add_argument("-num_classes", help="Number of the problem classes, e.g 2 for binary classification")



args = parser.parse_args()

"""
python3 utils/save_model_data.py -model_path '/home/smachar/Desktop/Internship/coding/Experiments/Decision Trees/Housing/dt_model_housing.pickle' 
    -to_store_thresholds '/home/smachar/Desktop/Internship/coding/Experiments/Decision Trees/Housing/Feature_thresholds.json' 
    -num_classes 4
"""


model_path_ = args.model_path.split('.')[-1]

with open(args.model_path, 'rb') as f:

    if model_path_ == 'pickle':
        dt_model = pickle.load(f)

    elif model_path_ == 'joblib':
        dt_model = joblib.load(f)

    else:
        sys.exit("The model must be saved either in pickle or joblib formats!")
    



class_names = [str(i) for i in range(int(args.num_classes))]



#fill the global variables with model's data recursively
recurse(model=dt_model, class_names=class_names, node_id=0)


with open(args.to_store_leaves, 'w') as f:
    json.dump(Leaves_Decisions, f)

with open(args.to_store_metadata, 'w') as f:
    json.dump(Leaves_Metada, f)

with open(args.to_store_thresholds, 'w') as f:
    json.dump(Feature_thresholds, f)

print('\nDONE!')



 
