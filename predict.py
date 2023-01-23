import argparse
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_curve, auc, roc_auc_score, make_scorer


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", default="lda", type=str)
parser.add_argument("--dataset", default="full", type=str)
args = parser.parse_args()
print(args)

# paths
if args.dataset == 'sample':
    results_path = "/home/devanshjain/topic_model/sample_pol_results"
    dataset_path = f"/home/devanshjain/topic_model/sample_pol"
else:
    results_path = "/home/devanshjain/topic_model/pol_twitter_results"
    dataset_path = f"/home/devanshjain/topic_model/pol_twitter"

outcomes_path = f"{dataset_path}/user_outcomes.csv"
prediction_path = f"{results_path}/{args.model_name}_results.csv"

vectors_dict = {
    'user_id': [],
    'user_vector': []
}

print("\n--- Creating features and outcomes ---")
print("--- Loading user vectors ---")
in_file = open(f"{results_path}/{args.model_name}_vectors.json", "r")
user_vectors = json.load(in_file)
for user, vector in user_vectors.items():
    vectors_dict['user_id'].append(user)
    vectors_dict['user_vector'].append(vector)

vectors_df = pd.DataFrame(vectors_dict)
vectors_df['user_id'] = vectors_df['user_id'].astype(int)
# print(vectors_df.head())
print("Shape of user vectors:", vectors_df.shape)

print("--- Loading outcomes ---")
outcomes_df = pd.read_csv(outcomes_path)
outcomes_df['user_id'] = outcomes_df['user_id'].astype(int)
# print(outcomes_df.head())
print("Shape of outcomes:", outcomes_df.shape)

print("--- Merging vectors and outcomes ---")
merged_df = vectors_df.merge(outcomes_df, how='left', on='user_id')
# print(merged_df.head())
print("Shape after merging:", merged_df.shape)

features = merged_df.user_vector.to_list()
outcome_age = merged_df.age.astype(int).to_list()
outcome_politics = merged_df.politics.astype(float).to_list()
outcome_gender = merged_df.gender.astype(int).to_list()

'''
Cross Validate and Metrics:
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
https://stackoverflow.com/questions/35876508/evaluate-multiple-scores-on-sklearn-cross-val-score
DLATK AUC: https://github.com/dlatk/dlatk/blob/b48b8b884835f6cdab3fad84b56d92b964046cb8/dlatk/classifyPredictor.py#L157
# https://stackoverflow.com/a/31161137 -> roc_auc=auc for normal predict(), i.e., predict_proba=False
RMSE negative: https://stackoverflow.com/a/27323356
'''

results_dict = {
    'Age_R2': [],
    'Age_RMSE': [],
    'Politics_R2': [],
    'Politics_RMSE': [],
    'Gender_AUC': []
}

print("\n----- Running prediction models -----")
print("--- Age: LinearRegression for R2 and RMSE ---")
age_reg = LinearRegression()
age_scores = cross_validate(age_reg, features, outcome_age, scoring=('r2', 'neg_root_mean_squared_error'), cv=5)
age_r2 = np.mean(age_scores['test_r2'])
results_dict['Age_R2'].append(age_r2)

age_rmse = -np.mean(age_scores['test_neg_root_mean_squared_error'])
results_dict['Age_RMSE'].append(age_rmse)
# print(age_scores)
# print("Age R2:", age_r2)
# print("Age RMSE:", age_rmse)

print("--- Politics: LinearRegression for R2 and RMSE ---")
politics_reg = LinearRegression()
politics_scores = cross_validate(politics_reg, features, outcome_politics, scoring=('r2', 'neg_root_mean_squared_error'), cv=5)
politics_r2 = np.mean(politics_scores['test_r2'])
results_dict['Politics_R2'].append(politics_r2)

politics_rmse = -np.mean(politics_scores['test_neg_root_mean_squared_error'])
results_dict['Politics_RMSE'].append(politics_rmse)
# print(politics_scores)
# print("Politics R2:", politics_r2)
# print("Politics RMSE:", politics_rmse)

print("--- Gender: LogisticRegression for ROC_AUC ---")
gender_reg = LogisticRegression()
gender_scores = cross_validate(gender_reg, features, outcome_gender, scoring='roc_auc', cv=5)
gender_auc = np.mean(gender_scores['test_score'])
results_dict['Gender_AUC'].append(gender_auc)
# print(gender_scores)
# print("Gender AUC:", gender_auc)

print("\n----- Results -----")
results_df = pd.DataFrame(results_dict)
print(results_df)
results_df.to_csv(prediction_path, index=False)