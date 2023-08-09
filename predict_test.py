"""
Regression with constant term only
https://stackoverflow.com/a/46916480
https://stackoverflow.com/a/66844483
"""

import argparse
import json
import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate

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

print("\n----- Running prediction models -----")
print("--- Age: LinearRegression for R2 and RMSE ---")
age_reg = DummyRegressor(strategy="mean")
features_dummy = np.ones((len(features), 1))
age_reg = age_reg.fit(features_dummy, outcome_age)
preds = age_reg.predict(features_dummy)
print(preds)
# age_scores = cross_validate(age_reg, features_dummy, outcome_age, scoring=('r2', 'neg_root_mean_squared_error'), cv=5)
score = age_reg.score(features_dummy, outcome_age)
print(score)