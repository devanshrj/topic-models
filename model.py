from octis.models.LDA import LDA
from octis.models.CTM import CTM
from dataset import Dataset

import argparse
import numpy as np
import pandas as pd
import time
import warnings
import json

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument("--model-name", default="lda", type=str)
parser.add_argument("--dataset", default="full", type=str)
args = parser.parse_args()
print(args)

# args
num_topics = 200

# model args
lda_args = {
    'alpha': 2,
    'iterations': 1000,
    'topk_words': 10
}

ctm_args = {
    'inference_type': 'combined',
    'model_type': 'prodLDA',
    'bert_model': 'princeton-nlp/sup-simcse-bert-base-uncased',
    'bert_path': '/home/devanshjain/topic_model/logs/ctm_combined'
}

# paths
if args.dataset == 'sample':
    results_path = "/home/devanshjain/topic_model/sample_pol_results"
    dataset_path = f"/home/devanshjain/topic_model/sample_pol"
else:
    results_path = "/home/devanshjain/topic_model/pol_twitter_results"
    dataset_path = f"/home/devanshjain/topic_model/pol_twitter"

# doc_topic_path = f"{results_path}/doctopics.txt"
topics_path = f"{results_path}/{args.model_name}_topics.txt"
state_path = f"{results_path}/{args.model_name}_distributions.csv"


# labels
labels_df = pd.read_csv(f"{dataset_path}/corpus.tsv", sep='\t', header=None)
labels_li = labels_df.iloc[:, 2].to_list()
# print(labels_li)

print("--- Dataset ---")
dataset = Dataset()
dataset.load_custom_dataset_from_folder(dataset_path)
print("Dataset initialised!")

print("--- Model ---")
if args.model_name == 'lda':
    model = LDA(num_topics=num_topics, alpha=lda_args['alpha'], iterations=lda_args['iterations'])
    model.partitioning(use_partitions=False)
    print("LDA initialised!")
elif args.model_name == 'ctm':
    model = CTM(num_topics=num_topics, model_type=ctm_args['model_type'],
            inference_type=ctm_args['inference_type'], bert_model=ctm_args['bert_model'],
            use_partitions=False, bert_path=ctm_args['bert_path'])
    print("CTM initialised!")


# train the model using default partitioning choice
print("Begin model training...")
t0 = time.time()
output = model.train_model(dataset)
t1 = time.time()
print(f"Model trained! Time taken = {t1 - t0:.4f}s")


print("Storing top words for each topic...")
f_topics = open(topics_path, "w")
if args.model_name == 'lda':
    topic_words = model._get_topics_words(lda_args['topk_words'])
    for idx, topic in enumerate(topic_words):
        top_words = f"|{idx}| " + ', '.join(topic)
        f_topics.write(top_words + '\n')
elif args.model_name == 'ctm':
    topic_words = output['topics']
    for idx, topic in enumerate(topic_words):
        if len(topic) > 10:
            topic = topic[:10]
        top_words = f"|{idx}| " + ', '.join(topic)
        f_topics.write(top_words + '\n')
print("Top words stored!")


# document-topic matrix
print("--- Topic document matrix ---")
topic_doc_m = output["topic-document-matrix"]
# octis gives topic-document matrix (topic, doc) but we need document-topic matrix -> transpose (https://stackoverflow.com/a/6473742)
doc_topic_m = [list(i) for i in zip(*topic_doc_m)]
print(len(doc_topic_m), len(doc_topic_m[0]))
print("Stored document-topic distributions!")


print("--- Creating user vectors ---")
topics_dict = {
    'user_id': [],
    'message_id': [],
    'topic': []
}

topic_dist = dict()

corpus_li = dataset.get_corpus()
labels_li = dataset.get_labels()
for doc_idx, (doc, label) in enumerate(zip(corpus_li, labels_li)):
    if doc_idx % 5000 == 0:  # progress update
        print(("Messages Read: %dk" % int(doc_idx/1000)))
    message_id = label.split("_")[0]
    user_id = label.split("_")[1]
    if user_id in topic_dist:
        topic_dist[user_id].append(np.array(doc_topic_m[doc_idx]))
    else:
        topic_dist[user_id] = [np.array(doc_topic_m[doc_idx])]
    # topics_dict["user_id"].append(user_id)
    # topics_dict["message_id"].append(message_id)
    # topics_dict["topic"].append(np.array(doc_topic_m[doc_idx]))
    # print(f"doc_idx: {doc_idx} \t doc: {doc} \t label: {label}")
    # print(f"user_id: {user_id} \t message_id: {message_id} \t topic: {np.array(doc_topic_m[doc_idx])}")


user_vectors = dict()
for user, docs in topic_dist.items():
    vector = np.mean(docs, axis=0).tolist()
    user_vectors[user] = vector


print(user_vectors['12488'])
out_file = open(f"{results_path}/{args.model_name}_vectors.json", "w")
json.dump(user_vectors, out_file)
out_file.close()
print("--- Generated topics and stored users vectors! ---")