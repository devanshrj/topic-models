"""
TO DO: remove characters like: \\u00a6, g\\u00e2, 'gâ' -> 183 such 1grams in the twitter vocabulary
"""

import ast
import os
import pandas as pd
import sqlalchemy
import string
import warnings

warnings.filterwarnings('ignore')

db = sqlalchemy.engine.url.URL(drivername='mysql',
                               host='127.0.0.1',
                               database='devanshjain',
                               query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})

engine = sqlalchemy.create_engine(db)

# paths
dataset_path = f"/home/devanshjain/topic_model/pol_twitter"
vocab_path = f"{dataset_path}/vocabulary.txt"
corpus_path = f"{dataset_path}/corpus.tsv"

# args
onegram_table = 'feat$1gram$twitter_pol_dedup$message_id$2en05'
tokenized_table = 'twitter_pol_dedup_tok'
n_stopwords = 100
group_id = 'user_id'
group_freq_thresh = 0
remove_non_ascii_chars = True

# get stopwords
print("--- Stop words ---")
query_stopw = f'''SELECT feat FROM {onegram_table} GROUP BY feat ORDER by sum(value) DESC limit {n_stopwords};'''
stopw_df = pd.read_sql(query_stopw, engine)
print("Stopwords stored in dataframe!")

stopwords_li = set(stopw_df.feat.tolist())
print("Stop words size:", len(stopwords_li))
print("Stop words list:", stopwords_li)


def remove_non_ascii(word):
    fixed = "".join(c for c in word if ord(c) < 128)
    return fixed


# get vocabulary
print("--- Vocabulary ---")
query_vocab = f'''SELECT distinct feat as vocabulary FROM {onegram_table}'''
vocab_df = pd.read_sql(query_vocab, engine)
print("Vocabulary stored in dataframe!")

init_vocab_li = set(vocab_df.vocabulary.tolist())
print("Vocabulary size (including stopwords):", len(init_vocab_li))

if remove_non_ascii_chars:
    fixed_vocabulary = [remove_non_ascii(word) for word in init_vocab_li]
    init_vocab_li = set(fixed_vocabulary)
    print("Vocabulary size (after removing weird characters):", len(init_vocab_li))

vocabulary_li = init_vocab_li - stopwords_li
final_vocab_df = pd.DataFrame(vocabulary_li)
print("Final vocabulary size:", len(vocabulary_li))


# preprocessing
def join_message(msg):
    msg_li = ast.literal_eval(msg)
    # https://www.geeksforgeeks.org/python-convert-a-string-representation-of-list-into-list/
    joined = ' '.join(msg_li)
    return joined


def preprocess_message(msg, vocabulary, lowercase=True, remove_punctuation=False):
    msg_p = msg
    msg_p = msg_p.replace('\n', ' ')
    msg_p = msg_p.replace('\t', ' ')
    if lowercase:
        msg_p = msg_p.lower()
    if remove_punctuation:
        msg_p = msg_p.translate(str.maketrans(
            string.punctuation, ' ' * len(string.punctuation)))
        msg_p = " ".join(msg_p.split())

    # filter words
    final_msg = [w for w in msg_p.split() if w in vocabulary]
    return ' '.join(final_msg)


print("--- Preprocessing ---")
# query_msgs = f'''SELECT {group_id}, message FROM {tokenized_table};'''
query_msgs = f'''SELECT message_id, user_id, message FROM {tokenized_table};'''
msgs_df = pd.read_sql(query_msgs, engine)

print("Joining tokenized messages...")
msgs_df['message'] = msgs_df.apply(
    lambda row: join_message(row['message']), axis=1)
print("Messages joined!")

print("Preprocessing messages...")
msgs_df['message'] = msgs_df.apply(
    lambda row: preprocess_message(row['message'], vocabulary_li), axis=1)
print("Messages preprocessed!")

if group_freq_thresh:
    print("Removing messages below group_freq_thres...")
    msgs_df = msgs_df[msgs_df['message'].apply(
        lambda msg: len(msg.split()) >= group_freq_thresh)]
    print("Removed messages below threshold!")

print("Assigning partiion...")
msgs_df['partition'] = 'train'
# msgs_df = msgs_df[['message', 'partition', f'{group_id}']]
msgs_df = msgs_df[['message', 'partition', 'message_id', 'user_id']]
print("Partition assigned!")

print("Removing NULL messages...")
msgs_df = msgs_df[msgs_df['message'].apply(lambda msg: len(msg) > 0)]
print("Removed NULL messages!")

# explicitly convert messages to strings and concatenate message_id and user_id
print("--- Converting messages to strings ---")
msgs_df['message'] = msgs_df['message'].astype(str)
msgs_df['label'] = msgs_df['message_id'].astype(str) + "_" + msgs_df['user_id'].astype(str)
msgs_df = msgs_df[['message', 'partition', 'label']]

print("--- Final dataset stats ---")
print(f"Number of messages: {msgs_df.shape[0]}")
print(f"Vocabulary size: {len(vocabulary_li)}")

print("Saving dataset...")
os.makedirs(dataset_path, exist_ok=True)
final_vocab_df.to_csv(vocab_path, sep='\n', header=False,
                      index=False, chunksize=50000)
msgs_df.to_csv(corpus_path, sep='\t', header=False,
               index=False, chunksize=50000)
print("Dataset saved!")
