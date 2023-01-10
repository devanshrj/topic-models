from octis.models.LDA import LDA
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset

import argparse
import csv
import numpy as np
import pandas as pd
import time
import warnings

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
    'bert_model': 'bert-base-nli-mean-tokens',
    'bert_path': '/home/devanshjain/topic_model/logs/ctm_combined'
}

# paths
if args.dataset == 'sample':
    parent_path = "/home/devanshjain/mlda/new"
    dataset_path = f"{parent_path}/processed_qualtrics"
else:
    results_path = "/home/devanshjain/pol_twitter_results"
    dataset_path = f"/home/devanshjain/topic_model/pol_twitter"

# doc_topic_path = f"{results_path}/doctopics.txt"
topics_path = f"{results_path}/{args.model_name}_topics.txt"
state_path = f"{results_path}/{args.model_name}_state"


# labels
labels_df = pd.read_csv(f"{dataset_path}/corpus.tsv", sep='\t', header=None)
labels_li = labels_df.iloc[:, 2].to_list()

print("--- Dataset ---")
dataset = Dataset()
dataset.load_custom_dataset_from_folder(dataset_path)
# print(dataset.__corpus)
print("Dataset initialised!")

# print("--- Model ---")
# if args.model_name == 'lda':
#     model = LDA(num_topics=num_topics, alpha=lda_args['alpha'], iterations=lda_args['iterations'])
#     model.partitioning(use_partitions=False)
#     print("LDA initialised!")
# elif args.model_name == 'ctm':
#     model = CTM(num_topics=num_topics, model_type=ctm_args['model_type'],
#             inference_type=ctm_args['inference_type'], bert_model=ctm_args['bert_model'],
#             use_partitions=False, bert_path=ctm_args['bert_path'])
#     print("CTM initialised!")


# # train the model using default partitioning choice
# print("Begin model training...")
# t0 = time.time()
# output = model.train_model(dataset)
# t1 = time.time()
# print(f"Model trained! Time taken = {t1 - t0:.4f}s")


# print("Storing top words for each topic...")
# f_topics = open(topics_path, "w")
# if args.model_name == 'lda':
#     topic_words = model._get_topics_words(lda_args['topk_words'])
#     for idx, topic in enumerate(topic_words):
#         top_words = f"|{idx}| " + ', '.join(topic)
#         f_topics.write(top_words + '\n')
# elif args.model_name == 'ctm':
#     topic_words = output['topics']
#     for idx, topic in enumerate(topic_words):
#         if len(topic) > 10:
#             topic = topic[:10]
#         top_words = f"|{idx}| " + ', '.join(topic)
#         f_topics.write(top_words + '\n')
# print("Top words stored!")


# # def print_doc_top_matrix(doc_topic_path, topic_doc_m, labels_li):
# #     csvWriter = csv.writer(open(doc_topic_path, 'w'))
# #     for label, topics in zip(labels_li, topic_doc_m):
# #         row = [label]
# #         row.extend(topics)
# #         csvWriter.writerow(row)


# # document-topic matrix
# print("--- Topic document matrix ---")
# topic_doc_m = output["topic-document-matrix"]
# # octis gives topic-document matrix (topic, doc) but we need document-topic matrix -> transpose (https://stackoverflow.com/a/6473742)
# doc_topic_m = [list(i) for i in zip(*topic_doc_m)]
# print(len(doc_topic_m), len(doc_topic_m[0]))
# # print_doc_top_matrix(doc_topic_path, topic_doc_m, labels_li)
# print("Stored document-topic distributions!")


# print("--- Assigning topics to documents ---")
# topics_dict = {
#     'user_id': [],
#     'message_id': [],
#     'topic': []
# }

# corpus_li = dataset.get_corpus()
# labels_li = dataset.get_labels()
# for doc_idx, (doc, label) in enumerate(zip(corpus_li, labels_li)):
#     if doc_idx % 5000 == 0:  # progress update
#         print(("Messages Read: %dk" % int(doc_idx/1000)))
#     message_id = label.split("_")[0]
#     user_id = label.split("_")[1]
#     topics_dict["user_id"].append(user_id)
#     topics_dict["message_id"].append(message_id)
#     topics_dict["topic"].append(np.array(doc_topic_m[doc_idx]))
#     print(
#         f"user_id: {user_id} \t message_id: {len(message_id)} \t topic: {np.array(doc_topic_m[doc_idx])}")
#     break

# topics_df = pd.DataFrame(topics_dict)
# topics_df.to_csv(state_path, sep=' ', header=False, index=False)
# print("--- Generated topics and stored messages encoded with topics! ---")