import argparse
import json
import pandas as pd
import sqlalchemy
from sqlalchemy.dialects import mysql

import warnings
warnings.filterwarnings('ignore')


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


print("\n--- Creating features and outcomes ---")
vectors_dict = {
    'id': [],
    'group_id': [],
    'feat': [],
    'value': [],
    'group_norm': []
}

print("--- Loading user vectors ---")
in_file = open(f"{results_path}/{args.model_name}_vectors.json", "r")
print(f"File name: {in_file}")
user_vectors = json.load(in_file)
id_count = 1
for user, vector in user_vectors.items():
    feat_count = 0
    for v in vector:
        vectors_dict['id'].append(id_count)
        vectors_dict['group_id'].append(user)
        vectors_dict['feat'].append(feat_count)
        vectors_dict['value'].append(v)
        vectors_dict['group_norm'].append(v)
        feat_count += 1
        id_count += 1


db = sqlalchemy.engine.url.URL(drivername='mysql',
                               host='127.0.0.1',
                               database='comp_topic_models',
                               query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})

engine = sqlalchemy.create_engine(db)


dtypes_dict = {
    'id': mysql.BIGINT(display_width=16, unsigned=True),
    'group_id': mysql.BIGINT(display_width=20),
    'feat': mysql.VARCHAR(length=12),
    'value': mysql.DOUBLE(),
    'group_norm': mysql.DOUBLE(),
}

print("--- Creating DataFrame of vectors ---")
vectors_df = pd.DataFrame(vectors_dict)
print(vectors_df.head())
print("Shape of user vectors:", vectors_df.shape)

print("--- Storing to MySQL ---")
table_name = f"{args.model_name}_user_vectors"
vectors_df.to_sql(table_name, engine, if_exists='replace', index=False, dtype=dtypes_dict, chunksize=50000)

print("--- Add keys ---")
with engine.connect() as con:
    con.execute(f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);")
    con.execute(f"ALTER TABLE {table_name} ADD KEY (group_id);")
    con.execute(f"ALTER TABLE {table_name} ADD KEY (feat);")