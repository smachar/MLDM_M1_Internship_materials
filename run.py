""" Imports: """

from ast import Import
import json
import joblib
import pickle
import numpy as np
import pandas as pd
import sys
import argparse
import os

from utils import metric_functions as mf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


""" GLOBAL VARIABLES: """

PRINT = False


""" code arguments handler: """
parser = argparse.ArgumentParser(description="Get the results of possible changes of one experiment:",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-model_path", help="File location to DT models (serialized)")
parser.add_argument("-data_file", help="File location to data")


parser.add_argument("-leaf_decisions", help="File where leaf nodes data are stored")
parser.add_argument("-leaf_metadata", help="File where leaf nodes metadata are stored")
parser.add_argument("-feature_thresholds", help="File where the feature thresholds/decisions are stored")
parser.add_argument("-num_classes", help="Number of the problem classes, e.g 2 for binary classification")
# parser.add_argument("-feature_names", help="List of feature names (with comma separating them), e.g bedrooms, bathrooms, sqft_living")
parser.add_argument("-folder", help="Folder where to store the results")
parser.add_argument("-print", help="True/False value in order to print some info")



# parser.add_argument("-c","--class_names", nargs='+', const=None, help="the names of model classes if known, e.g -c 1 2 3")

args = parser.parse_args()

if args.print=='True':
    PRINT = True


""" Read data: """

#read the leaf nodes' decisions wher every node has a list of decisions/inequalities (which themselves are encoded as list of shape [min max])
with open(args.leaf_decisions, 'rb') as f:
    leaf_decisions = json.load(f)


#read leaf nodes metadat; every node contains data such as impurity value, number of samples ...
with open(args.leaf_metadata, 'rb') as f:
    leaf_metadata = json.load(f)


#read all model's thresholds (values)
with open(args.feature_thresholds, 'rb') as f:
    feature_thresholds = json.load(f)


#read the Decision Tree saved model
model_path_ = args.model_path.split('.')[-1]
with open(args.model_path, 'rb') as f:

    if model_path_ == 'pickle':
        dt_model = pickle.load(f)

    elif model_path_ == 'joblib':
        dt_model = joblib.load(f)

    else:
        sys.exit("The model must be saved either as pickle or joblib formats!")


""" Initializations: """


# feature_names = args.feature_names
feature_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'statezip']

class_names = [str(i) for i in range(int(args.num_classes))]
# class_names = ['0', '1', '2']

#ids of all leaf nodes
leaves = leaf_decisions.keys()

#ids of leaf nodes of 0 impurity
leaves_pure = [i for i in leaves if leaf_metadata[str(i)]['impurity']==0]

#a dictionary with class names (0,1,2) as keys and their associated leaf node's ids
class_leaves = dict()

#same as class_leaves containing only nodes of 0 impurity
class_leaves_pure = dict()

if PRINT:
    print("Number of leaf nodes and pure leaf nodes in every class:")
for j in class_names:
    class_leaves[j] = [i for i in leaves if leaf_metadata[str(i)]['class']==j]
    class_leaves_pure[j] = [i for i in leaves if leaf_metadata[str(i)]['class']==j and leaf_metadata[str(i)]['impurity']==0]
    if PRINT:
        print("class", j, ": ", len(class_leaves[j]), f"({len(class_leaves_pure[j])} pure leaves)")

if PRINT:
    print()

# features importance
if PRINT:
    temp = dt_model.tree_.compute_feature_importances()
    temp2 = list(reversed(np.argsort(dt_model.tree_.compute_feature_importances())))
    for i in temp2:
        print(i, feature_names[i], round(temp[i], 3))



# read original data
data = pd.read_csv(args.data_file, index_col=0)
data = data.reset_index()[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
       'yr_built', 'yr_renovated', 'statezip', 'output']]

#change statezip column to only zipcode, e.g WA 98133 ==> 98133
data['statezip'] = data['statezip'].apply(lambda x: int(x.split()[1]))



#get suggested possible changes

obs_changes_df, obs_variables = mf.get_observation_possible_changes(
    data=data, model=dt_model, dataLen=len(data), num_features=len(feature_names), class_leaves=class_leaves,
    leaf_decisions=leaf_decisions, class_names=class_names, feature_names=feature_names)



"""
    In this first part the sort is only fucosing on the target node thresholds.
    
    target node: the possible go-to leaf in order to correct the classification,
    thresholds: the must be satisfied values
"""

#calculate the score of every change including multiple feature changes

obs_changes_df['score'] = obs_changes_df.apply(mf.calculate_change_score, axis=1, args=([['Must Satisfy']]))


#create partitioned dataframe based on number of features in every change

obs_df_parts = mf.partition_obs_df(observation_changes_df=obs_changes_df)
if PRINT:
    print("Number of change groups for only target node thresholds: ")
    print(obs_df_parts.keys())


""" work on one feature changes: """

#order changes by value of score

counts = mf.count_trues(obs_df_parts['change_1_features'].duplicated(subset=['Feature name', 'score']).to_list())
change_1_features = obs_df_parts['change_1_features'].drop_duplicates(subset=['Feature name', 'score'])
change_1_features.loc[:, 'suggested times'] = counts
change_1_features.sort_values(by=["score"], inplace=True)


#aggregate on individual features by mean of their score values to get
#a list of recommended features based on the mean of every feature's score values

change_1_features_aggregated = obs_df_parts['change_1_features'].groupby(["Feature name"]).agg(average_score=('score', 'mean'), 
                                        suggested_times=('score', 'count')).sort_values(by="average_score")

""" work on multiple-feature changes: """

#sort all multi-feature changes based on the sum of score values of their different features,
# so that changes with small sum of score values are presented at top.

for i in range(2, len(obs_df_parts)+1):
    boolean_list, counts = mf.count_duplicates(obs_df_parts[f"change_{i}_features"], i)
    obs_df_parts[f"change_{i}_features"] = obs_df_parts[f"change_{i}_features"][boolean_list]
    obs_df_parts[f"change_{i}_features"].loc[:, 'suggested times'] = counts
    obs_df_parts[f"change_{i}_features"] = mf.sort_multi_feature_df(obs_df_parts[f"change_{i}_features"], i, column_name='score')



"""
    In this first part the sort is fucosing on all the Decision Tree model thresholds.
    
"""

#in this df the score-all is calculated for every change (row) using all thresholds of the row feature

obs_changes_df_all = obs_changes_df[['Feature name', 'Value', 'Satisfied', 'Must Satisfy']]
obs_changes_df_all['score-all'] = obs_changes_df_all.apply(mf.calculate_change_score_all, axis=1, args=([feature_thresholds]))
 

#create partitioned dataframe based on number of features in every change

obs_df_parts_all = mf.partition_obs_df(observation_changes_df=obs_changes_df_all)
if PRINT:
    print("Number of change groups for all DT thresholds: ")
    print(obs_df_parts_all.keys())


""" sort on one feature changes: """

counts = mf.count_trues(obs_df_parts_all['change_1_features'].duplicated(subset=['Feature name', 'score-all']).to_list())
obs_df_parts_all['change_1_features'] = obs_df_parts_all['change_1_features'].drop_duplicates(subset=['Feature name', 'score-all'])
obs_df_parts_all['change_1_features'].loc[:, 'suggested times'] = counts
obs_df_parts_all['change_1_features'] = obs_df_parts_all['change_1_features'].sort_values(by=["score-all"])


""" sort on multiple-feature changes: """

for i in range(2, len(obs_df_parts_all)+1):
    boolean_list, counts = mf.count_duplicates(obs_df_parts_all[f"change_{i}_features"], i)
    obs_df_parts_all[f"change_{i}_features"] = obs_df_parts_all[f"change_{i}_features"][boolean_list]
    obs_df_parts_all[f"change_{i}_features"].loc[:, 'suggested times'] = counts
    obs_df_parts_all[f"change_{i}_features"] = mf.sort_multi_feature_df(obs_df_parts_all[f"change_{i}_features"], i, column_name='score-all')


"""
    save all data
"""

#create an experiment folder
parent_path = os.path.join(args.folder, str(len(os.listdir(args.folder))+1))
some_decisions_path = os.path.join(parent_path, "target_node_decisions")
all_decisions_path = os.path.join(parent_path, "all_decisions")

os.mkdir(parent_path)
os.mkdir(some_decisions_path)
os.mkdir(all_decisions_path)

#save data related to using only target node thresholds

obs_changes_df.to_csv(os.path.join(some_decisions_path, 'changes.csv'))
change_1_features.to_csv(os.path.join(some_decisions_path, 'change_1_features.csv'))
change_1_features_aggregated.to_csv(os.path.join(some_decisions_path, 'change_1_features_aggregated.csv'))

for i in range(2, len(obs_df_parts)+1):
    obs_df_parts[f"change_{i}_features"].to_csv(os.path.join(some_decisions_path, f"change_{i}_features.csv"))


#save data related to using all dt thresholds

obs_changes_df_all.to_csv(os.path.join(all_decisions_path, 'changes.csv'))

for i in range(1, len(obs_df_parts)+1):
    obs_df_parts_all[f"change_{i}_features"].to_csv(os.path.join(all_decisions_path, f"change_{i}_features.csv"))

#save related experiment data
with open(os.path.join(parent_path, "metadata.json"), 'w') as f:
     f.write(json.dumps(obs_variables)) # json.loads


print('\nDONE!')

"""
python3 run.py -model_path './Housing/dt_model_housing.pickle' -data_file '../Datasets/housing_classif.csv' -leaf_decisions './Housing/leaf_decisions.json' 
-leaf_metadata './Housing/leaf_metadata.json' -feature_thresholds './Housing/feature_thresholds.json' -num_classes 3 -folder './Housing/Experiments' -print 'True'

"""
