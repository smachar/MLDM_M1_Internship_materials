#import
import numpy as np
import pandas as pd
from collections import Counter


#function definitions

def get_pure_class_samples(class_name, leaf_metadata, class_leaves):
    """
    Returns a list of the number of samples in every leaf node of the class class_name

    Parameters
    ----------
    class_name : the name of the considered classification/prediction class (str)
    
    leaf_metadata : the leaf nodes metadata where the number of samples 
        is included (dict)

    class_leaves : dictionary of class names as keys and their corresponding 
        nodes (IDs) as values (dict)

    """

    return [leaf_metadata[i]['samples'] for i in class_leaves[class_name]]


def get_suspected_features_freqs(observation, class_leaves, leaf_decisions, class_names):

    """
    Return the suspected features - those with values (in observation) not satisfied by the leaf node inequality-, 
        and their corresponding frequencies across all nodes of different classes

    Parameters
    ----------
    observation : the sample used for predicition (np.array)

    class_leaves : dictionary of class names as keys and their corresponding 
        nodes (IDs) as values (dict)
    
    leaf_decisions : the leaf nodes inequalities/decisions (dict)

    class_names : classification/prediction class names (list)

    """

    suspected_ft = dict()
    """
    dictionary of suspected/not-satisfied features of every class's leaf nodes
    e.g {'0': {'6': ['2', '3'],  '9': ['2', '3', '8'] } } : leaf nodes 6 and 9 belonging to class 0 have different features 
    that were not satisfied by the observation's values. Satisfaction holds when observation's value at same feature 
    is in the feature's decision/inequality [min max] interval.
    """

    feature_freq = dict()
    """
    dictionary of every class feature frequencies.
    e.g {'0': {'2': 70, '3': 48, '12': 60, '8': 26, '1': 16 .. } } : there are 70 leaf nodes of
    a total of 333 labelled as class 0 that are not respected/satisfied by the observation's
    values.
    """

    for class_ in class_names:
        suspected_ft[class_] = dict()
        feature_freq[class_] = dict()

        for leaf in class_leaves[class_]:
            temp = []
            for feature in leaf_decisions[leaf]:
                if feature not in list(feature_freq[class_].keys()):
                    feature_freq[class_][feature] = 0

                ft_decision = leaf_decisions[leaf][feature]
                
                # print(leaf, feature, ft_decision, observation1[0][int(feature)])
                if ft_decision[0] is not None and observation[0][int(feature)] < ft_decision[0]:
                    temp.append(feature)
                    feature_freq[class_][feature]+=1
                    continue
                elif ft_decision[1] is not None and observation[0][int(feature)] > ft_decision[1]:
                    temp.append(feature)
                    feature_freq[class_][feature]+=1
                    continue
                
            suspected_ft[class_][leaf] = temp
    return suspected_ft, feature_freq


def create_observation_df(suspected_ft, observation, pred_path, actual_class, feature_names, leaf_decisions):

    """"
    Create a DataFrame of all possible changes to be done to the values of observation in order for it to be well classified
    and then to detect a data quality issue with data value.

    The DF is order based on number of features to be changed; e.g changes that require one feature 
        modification are presented at the top of DF

    Parameters
    ----------
    suspected_ft : suspected features (dict)

    observation : observation values (np.array)

    pred_path : nodes used for the prediction (list)

    actual_class : the name of the ground truth class name (str)

    feature_names : the names of features (list)

    leaf_decisions : the leaf nodes inequalities/decisions (dict)
    """

    #get features of the predicted//fall in leaf node (those are the satisfied features of this leaf node)
    pred_leaf_features = list(leaf_decisions[str(pred_path[-1])].keys())

    tuples = []
    data = []
    for node, features in suspected_ft[str(actual_class)].items():
        #work only on node's features that are in leaf features
        for feature in features:
            if feature in pred_leaf_features:
                tuples.append((node, feature))
                #append to the df the feature name, current value (detected 'error'), not violated inequality and 
                #must satisfy inequality  
                data.append([feature_names[int(feature)], observation[0][int(feature)], 
                leaf_decisions[str(pred_path[-1])][feature], leaf_decisions[node][feature]])

    index = pd.MultiIndex.from_tuples(tuples, names=["Node", "Feature"])
    observation_df = pd.DataFrame(data, index=index, columns=["Feature name", "Value", "Satisfied", "Must Satisfy"])

    #sort the df based on number of features that must changed in order for the observation to fall in that leaf node
    index_sorter = observation_df.groupby(['Node']).count().sort_values(['Feature name'], ascending=True).index
    index_to_sort = observation_df.index.get_level_values(0)
    return observation_df.iloc[index_to_sort.reindex(index_sorter)[1]]


def get_observation(data, idx, change_idx, value):
    """
    Get an instance from the data at index 'idx', 
    Change its feature at index 'change_idx' to another value of 'value' 
    Return new observation along with its actual class (ground truth)

    Parameters
    ----------
    data : the samples data (pd.DataFrame)

    idx : index of target sample/instance (int)

    change_idx : the target (to be changed) feature index (int)

    value : new value of the targer feature (float)
    """
    
    observation = list(data.loc[idx, :])[:-1]
    actual_class = int(data.loc[idx, ['output']])
    observation[change_idx] = value
    return np.array(observation, dtype=np.float32).reshape(1, -1), actual_class


def count_trues(boolean_list):

    """
    Count number of successive TRUE values in a boolean list

    Parameters
    ----------
    boolean_list: list of TRUE and FALSE as elements (list)

    Return
    ----------
    a list of number of successive TRUE values in the boolean list

    """

    counts = []
    c=1
    for i in boolean_list[1:]:
        if i==True:
            c+=1
        else:
            # print(i, c)
            counts.append(c)
            c=1
            
    counts.append(c)
    return counts



#dataframce manipulation functions

def calculate_change_score(x, col_names):

    """
    To calculate the score of every changes in the recommended possible
    changes dataframe.

    score is a positive value of the division of the observation's current value 
    and its closest value of the set of values extracted from the col_names list
    of columns.

    This function is used with the pandas apply function.

    Parameters
    ----------
    x: the row of the dataframe (pd.Serires)

    col_name: list of columns to extract their values that will be used in comparison
    with observation's current value in order to calculate score (list of str)

    Return
    ----------
    the absulote value of score which represents the degree of importance of the data
    frame row; aka the recommended change.
	
    
    The more the score's value is close to zero, the more confidance that there is a 
    data quality issue with that fearure's value of the observation

    """

    obs_value = x.Value
        
    values = []
    for i in x[col_names]:
        values += i

    values = np.array([i for i in values if i!=None])

    closest_val = values[np.abs(values - obs_value).argmin()]

    if closest_val<obs_value:
        score = closest_val/obs_value
    else:
        score = obs_value/closest_val

    return abs(score)


def partition_obs_df(observation_changes_df):

    """
    Partition the observation's dataframe into several DFs based on the number 
    of features that must be changed in order for the observation to be in the correct class

    Parameters
    ----------
    observation_changes_df: the created dataframe from a given observation values/ df 
    of possible changes to every value of the observation (pd.DataFrame)

    return
    ----------
    A dictionary, obs_df_all, of keys in the format of "change_(number of features in the suggested change)_features"

    """
    counts = list(Counter(Counter(observation_changes_df.index.get_level_values(0)).values()).values())

    boundries = []
    to_idx = 0
    for i in range(len(counts)):
        from_idx = to_idx
        to_idx = from_idx+counts[i]*(i+1)
        boundries.append((from_idx, to_idx))

    obs_df_all = {}
    for i in range(len(boundries)):
        var_name = f"change_{i+1}_features"
        obs_df_all[var_name] = observation_changes_df[boundries[i][0]:boundries[i][1]].copy()
    
    return obs_df_all


def sort_multi_feature_df(dataframe, num_chg_features, column_name='score'):

    """
    Sort a dataframe of suggested changes of more than one feature to be changed.

    The sort is based on the sum of score values of different features withing 
    the same suggested change. Changes with small sum of scores' values are presented at top.

    Parameters
    ----------
    dataframe: the to-be-sorted dataframe (pd.DataFrame)

    num_chg_features: the number of features to be changed within every suggested change (int)

    column_name: is the column where the values of the score are stored (str)

    Return
    ----------
    A new sorted dataframe with an extra column of "Total Score"; the sum up scores' 
    of features belonging to the same suggested change
    """

    #add up the scores
    added_scores = dataframe.groupby(level=[0], sort=False).sum()[column_name].sort_values().values

    #duplicate argsort indices based on num_chg_features
    indices = []
    for i in dataframe.groupby(level=[0], sort=False).sum()[column_name].argsort().values:
        indices.append(i*num_chg_features)
        for j in range(1, num_chg_features):
            indices.append(i*num_chg_features+j)

    #get original index of the df
    org_indices = list(dataframe.index)

    #sort the original index using argsort indices
    tuples = []
    for i in np.array(org_indices)[indices]:
        tuples.append((i[0], i[1]))


    #reindex the dataframe so it becomes sorted 
    dataframe = dataframe.reindex(pd.MultiIndex.from_tuples(tuples, names=('Node', 'Feature')))

    #duplicate scores to add them to the df as new column
    added_scores_ = []
    for i in added_scores:
        for j in range(num_chg_features):
            added_scores_.append(i)
        
    dataframe["Total score"] = added_scores_

    return dataframe


def count_duplicates(observation_changes_df, num_chg_features):

    """
    Count the number of duplicated changes in the observation_changes_df
    (every change is a combination of a number of features)

    Parameters
    ----------
    observation_changes_df: a dataframe of recommended changes of more than one feature (pd.DataFrame)

    num_chg_features: the number of features within every changes (int)

    Return
    ----------
    boolean_list: a lis of TRUE and FALSE elements indecating which row to keep (TRUE) and which to
    drop (FALSE)

    counts_: a list of number of duplicated changes 
    (repeted depending on num_chg_features because it will be appended to the df)
    
    """

    temp = observation_changes_df['Feature name'].to_list()
    boolean_list = [True for _ in range(num_chg_features)]
    unique_groups = [[temp[i] for i in range(num_chg_features)]]
    counts = [1]

    for i in range(num_chg_features, len(temp), num_chg_features):

        to_keep = True
        current_grp = sorted([temp[i+j] for j in range(num_chg_features)])

        for k in range(len(unique_groups)):
            if current_grp == sorted(unique_groups[k]):
                counts[k] += 1
                to_keep = False
                break

        
        boolean_list.extend([to_keep for _ in range(num_chg_features)])

        if to_keep==True:
            unique_groups.append(current_grp)
            counts.append(1)

    counts_ = []
    for i in counts:
        for j in range(num_chg_features):
            counts_.append(i)

    return boolean_list, counts_


def calculate_change_score_all(x, feature_thresholds):
    """
    To calculate the score-all of every changes in the recommended possible
    changes dataframe.

    score-all is a positive value of the devision of the observation's (feature's) current value 
    and its closest value of the set of all values in the DecisionTree of the same feature.

    This function is used with the pandas apply function.

    Parameters
    ----------
    x: the row of the dataframe (pd.Serires)

    feature_thresholds: the list of all feature's values in the DecisionTree (list)

    Return
    ----------
    the absulote value of score-all which represents the degree of importance of the data
    frame row; aka the recommended change.

    The more the score-all's value is close to zero, the more confidance that there is a 
    data quality issue with that fearure's value of the observation

    """
    
    obs_value = x.Value
    obs_feature = x.name[1]
            
    closest_val = feature_thresholds[obs_feature][np.abs(np.array(feature_thresholds[obs_feature])-obs_value).argmin()]

    if closest_val<obs_value:
        beta = closest_val/obs_value
    else:
        beta = obs_value/closest_val

    return abs(beta)



#run experiment 

def run_rdm_expt(data, model, dataLen, num_features):

    """
    Run one experiment by taking a random data point, random feature (column),
    and then randomly changing its value.

    Parameters
    ----------
    data: the data (pd.DataFrame)

    model: the DecisionTree model (sklearn.tree._classes.DecisionTreeClassifier)

    dataLen: number of data points (int)

    num_features: number of features (int)

    Return
    ----------
    the chosen data point's: actual class, predicted class before implementing the change,
    predicted class after the change, the data point and feature randomly chosen indices,
    and finally the observation's values (row with the edited feature's value)

    """

    idx =  np.random.randint(0, dataLen) #4600
    change_idx = np.random.randint(0, num_features) #13
    value = 100*np.random.random_sample()+100

    #less important features: 4 floors, 6 view, 5 waterfront for house price classification problem

    obs, actual_class = get_observation(data, idx, change_idx, value)
    prd_class_before_chng = model.predict(np.array(list(data.loc[1359, :])[:-1], dtype=np.float32).reshape(1, -1), check_input=True)[0]
    prd_class = model.predict(obs, check_input=True)[0]

    return actual_class, prd_class_before_chng, prd_class, change_idx, idx, obs


def get_observation_possible_changes(data, model, dataLen, num_features, class_leaves, leaf_decisions, class_names, feature_names):

    """
    Run multinple experiments to get a 'valuable experiment' by picking random data points and calculate their prediction before 
    and after implementing the feature change. if the predicted the class before change is exactly
    the same as the actual class (to not fall onto false predictions), and different than the predicted
    class after the change then this is this experiment is a 'valuable experiment'.

    the change here is the change of the value of a random feature of the randomly picked data point.

    Parameters
    ----------
    data: the data (pd.DataFrame)

    model: the DecisionTree model (sklearn.tree._classes.DecisionTreeClassifier)

    dataLen: number of data points (int)

    num_features: number of features (int)

    class_leaves : dictionary of class names as keys and their corresponding 
    nodes (IDs) as values (dict)
    
    leaf_decisions : the leaf nodes inequalities/decisions (dict)

    class_names : classification/prediction class names (list)

    feature_names : classification/prediction feature names (list)

    Return
    ----------
    the dataframe of all possible changes that could correct the wrong classification by 
    handling the suggested feature's value (the identified data/feature quality issue)

    and the experiment related variables/information

    """

    #run multinple experiments to get the 'valuable' one
    while True:

        actual_class, prd_class_before_chng, prd_class, change_idx, idx, obs = run_rdm_expt(data, model, dataLen, num_features)

        if actual_class==prd_class_before_chng and prd_class!=prd_class_before_chng:

            # print("Original observation:") 
            # display(data[idx:idx+1])
            # print("Edited observation: ", list(obs[0]))
            # print(f"Change at {change_idx, feature_names[change_idx]} from {data.loc[idx,data.columns[change_idx]]} to {obs[0][change_idx]}")
            # print(f"Actual class (data) ==> {actual_class}")
            # print(f"Predicted class before change ==> {prd_class_before_chng}")
            # print("Predicted class after change ==> ", prd_class)
            
            node_indicator = model.decision_path(obs, check_input=True)
            sample_id = 0
            pred_path = node_indicator.indices[
                node_indicator.indptr[sample_id] : node_indicator.indptr[sample_id + 1]
            ]
            # print("Prediction path:", pred_path)

            # print("leaf decisions: ", leaf_decisions[str(pred_path[-1])])

            suspected_ft, feature_freq = get_suspected_features_freqs(observation=obs, 
                                        class_leaves=class_leaves, leaf_decisions=leaf_decisions, class_names=class_names)

            # print("features frequencies of nodes of the actual class:", feature_freq[str(actual_class)])

            obs_changes_df = create_observation_df(suspected_ft=suspected_ft, observation=obs, 
                    pred_path=pred_path, actual_class=actual_class, feature_names=feature_names, leaf_decisions=leaf_decisions)
            # print("\nDf of all possible changes ordered by number of features to be changed: ")
            # display(obs_changes_df)
            obs_variables = {
                "original_data": str(list(data[idx:idx+1].values[0])), 
                "edited_data": str(list(obs[0])),
                "feature_chg_idx": str(feature_names[change_idx]),
                "chg_from": str(data.loc[idx,data.columns[change_idx]]),
                "chg_to": str(obs[0][change_idx]),
                "actual_class": str(actual_class),
                "predicted_before": str(prd_class_before_chng),
                "predicted_after": str(prd_class),
                "prediction_path": str(pred_path),

            }
            break
        
    return obs_changes_df, obs_variables

