import argparse
import json
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

print("--- Loading user vectors ---")
in_file = open(f"{results_path}/{args.model_name}_vectors.json", "r")
user_vectors = json.load(in_file)
for user, vector in user_vectors.items():
    vectors_dict['user_id'].append(user)
    vectors_dict['user_vector'].append(vector)

vectors_df = pd.DataFrame(vectors_dict)
vectors_df['user_id'] = vectors_df['user_id'].astype(int)
print(vectors_df.head())
print(vectors_df.shape)

print("--- Loading outcomes ---")
outcomes_df = pd.read_csv(outcomes_path)
outcomes_df['user_id'] = outcomes_df['user_id'].astype(int)
print(outcomes_df.head())
print(outcomes_df.shape)

print("--- Merging vectors and outcomes ---")
merged_df = vectors_df.merge(outcomes_df, how='left', on='user_id')
print(merged_df.head())
print(merged_df.shape)

print("--- Creating features and outcomes ---")
features = merged_df.user_vector.to_list()
outcome_age = merged_df.age.astype(int).to_list()
outcome_politics = merged_df.politics.astype(float).to_list()
outcome_gender = merged_df.gender.astype(int).to_list()

'''
Cross Validate and Metrics:
https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
https://stackoverflow.com/questions/35876508/evaluate-multiple-scores-on-sklearn-cross-val-score
DLATK AUC: https://github.com/dlatk/dlatk/blob/b48b8b884835f6cdab3fad84b56d92b964046cb8/dlatk/classifyPredictor.py#L157
'''

print("--- Age: LinearRegression for R2 and RMSE ---")
age_reg = LinearRegression()
age_scores = cross_validate(age_reg, features, outcome_age, scoring=('r2', 'neg_root_mean_squared_error'), cv=5)
print(age_scores)

print("--- Politics: LinearRegression for R2 and RMSE ---")
politics_reg = LinearRegression()
politics_score = cross_validate(politics_reg, features, outcome_age, scoring=('r2', 'neg_root_mean_squared_error'), cv=5)
print(politics_score)

print("--- Gender: LogisticRegression for ROC_AUC ---")
gender_reg = LogisticRegression()
gender_scores = cross_validate(gender_reg, features, outcome_gender, scoring='roc_auc', cv=5)
print(gender_scores)

