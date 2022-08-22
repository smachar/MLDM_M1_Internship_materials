#imports

import numpy as np

#import the modified version of sklearn.tree export class
from . import export_me as me


# from sklearn.externals.six import 
from six import StringIO  
from IPython.display import Image
import pydotplus



def plot_tree(model, feature_names, class_names, file_name=None, 
                start_node_id=0, show=False, path_me=[]):
    
    """
    Colored diagram of the DT model to plot or save (or both), 
    using only a predefined set of nodes (e.g: visualizing the path of one prediction, ...)

    Parameters
    ----------
    model : the DT classifier (sklearn.tree._classes.DecisionTreeClassifier)

    feature_names : the features names (list)

    class_names : names of prediction classes (list)

    file_name : a location where to store the diagram (string:path)

    start_node_id : the starting node usually is the root (default 0)
        but could anything else except the leaf nodes (int)
    
    show : either to dsiplay or not the digram (useful in notebook) (deafult False) (boolean)

    path_me : a list of nodes to be inlcluded in the diagram (e.g: a prediction path 
        nodes) (default []: plotting all nodes)
    """

    
    dot_data = StringIO()

    me.export_graphviz(model, out_file=dot_data,  filled=True, rounded=True,
        special_characters=True, feature_names = feature_names, class_names=class_names, 
        node_ids=True, start_node_id=start_node_id, path_me=path_me)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    #save
    if file_name:
        graph.write_png(file_name)
        
    #display (useful when used with notebook)
    if show:
        return Image(graph.create_png())
    return None


def lead_to_actual(dt_model, actual_val, node=0):
    """
    Search the tree structure for paths that could lead to the actual prediction (actual_val) 
    from any given node (node).
    e.g: from node 0, what are the possible paths that could lead to the actual class?

    Parameters
    ----------
    dt_model : the DT classifier (sklearn.tree._classes.DecisionTreeClassifier)

    node : the starting node (default 0 : root) anything but leaf nodes

    actual_val : the ground truth classification/prediction


    Return
    ----------
    return list of possible paths (nodes IDs)
    """

    left_node = dt_model.tree_.children_left[node]
    right_node = dt_model.tree_.children_right[node]

    # both of left and right nodes are not leaf nodes
    if dt_model.tree_.children_left[left_node]!=-1 and dt_model.tree_.children_left[right_node]!=-1:
        path = []
        left_trace = lead_to_actual(dt_model, left_node, actual_val)
        right_trace = lead_to_actual(dt_model, right_node, actual_val)
        if len(left_trace)!=0:
            for i in left_trace:
                path.append([node]+i)

        if len(right_trace)!=0:
            for i in right_trace:
                path.append([node]+i)
        
        return path
        # return [(node, left_node)] + lead_to_actual(left_node, actual_val) + [(node, right_node)] + lead_to_actual(right_node, actual_val)

    preds = []
        
    # check if left_node is a leaf node
    if dt_model.tree_.children_left[left_node]==-1:
        temp = dt_model.tree_.value[left_node][0]
        pred_leaf = np.argmax(temp/sum(temp))
        # print(pred_leaf)

        if pred_leaf==actual_val:
            #can reach to actual from from the node
            #possible paths
            preds.append([node, left_node])
            # preds+=[node, left_node]

        #right_node is a leaf node
        # if right_node==-1:
        if dt_model.tree_.children_left[right_node]==-1:
            temp = dt_model.tree_.value[right_node][0]
            pred_leaf = np.argmax(temp/sum(temp))
            # print(pred_leaf)

            if pred_leaf==actual_val:
                preds.append([node, right_node])
                # preds+=[node, right_node]
            return preds

        #when right_node is not a leaf node
        else:
            right_trace = lead_to_actual(dt_model, right_node, actual_val)
            if len(right_trace)!=0:
                for i in right_trace:
                    preds.append([node]+i)
            return preds

    #left_node is not a leaf node
    else:
        # preds.append(lead_to_actual(left_node, actual_val))
        left_trace = lead_to_actual(dt_model, left_node, actual_val)
        if len(left_trace)!=0:
            for i in left_trace:
                preds.append([node]+i)

        #check if right_node is a leaf node
        if dt_model.tree_.children_left[right_node]==-1:
            temp = dt_model.tree_.value[right_node][0]
            pred_leaf = np.argmax(temp/sum(temp))

            if pred_leaf==actual_val:
                preds.append([node, right_node])
            return preds

        #when right_node is not a leaf node
        else:
            right_trace = lead_to_actual(dt_model, right_node, actual_val)
            if len(right_trace)!=0:
                for i in right_trace:
                    preds.append([node]+i)
            return preds



    