import argparse
import pandas as pd
import sqlalchemy
import warnings

warnings.filterwarnings('ignore')


def topic_uniqueness(topics_dict, L, renorm=False):
    '''
        topics_dict (dictionary): topic-id -> word -> weight
        L (int): number of top words to consider
        renorm (boolean): renormalize topic uniqueness to between 0 and 1, 
                          otherwise topic uniqueness is between 1/len(topics_dict) and 1
    '''
    K = len(topics_dict)
    if L == 0 or K == 0:
        print("Both L and K must be non-zero")
        return None, None
    
    # sort the dict
    sorted_topics_dict = dict()
    for topic, words in topics_dict.items():
        sorted_words = [k for k, v in sorted(words.items(), key=lambda item: item[1], reverse=True)][0:L]
        sorted_topics_dict[topic] = {word: words[word] for word in sorted_words}
    
    topic_scores = dict()
    for topic, words in sorted_topics_dict.items():
        this_score = 0
        for word in words:
            cnt_l_k = 0
            for topic_inner, words_inner in sorted_topics_dict.items():
                if word in words_inner:
                    cnt_l_k += 1
            this_score += 1/float(cnt_l_k)
        topic_scores[topic] = this_score/float(L)

    TU = sum([v for k,v in topic_scores.items()])/float(K)
    if renorm:
        TU = (TU-1/float(K)) / float((1-1/float(K)))
    return TU, topic_scores


parser = argparse.ArgumentParser()
parser.add_argument("--database", default="Reddit_Depression_and_India", type=str)
parser.add_argument("--lexicon", default="both_msgs_200_freq_t50ll", type=str)
parser.add_argument("--l", default=10, type=int)
args = parser.parse_args()
print(args)


db = sqlalchemy.engine.url.URL(drivername='mysql',
                               host='127.0.0.1',
                               database=args.database,
                               query={'read_default_file': '~/.my.cnf', 'charset': 'utf8mb4'})

engine = sqlalchemy.create_engine(db)


print("--- Reading topic lexicon ---")
query = f'''SELECT * FROM {args.lexicon};'''
lexicon_df = pd.read_sql(query, engine)
print(lexicon_df)


print("--- Generating topics dict ---")
topics_dict = dict()
for index, row in lexicon_df.iterrows():
    topic_id = row['category']
    term = row['term']
    weight = row['weight']
    if topic_id in topics_dict:
        topics_dict[topic_id][term] = weight
    else:
        topics_dict[topic_id] = dict()
        topics_dict[topic_id][term] = weight
print("--- Topics dict generated! ---")


TU, topic_scores = topic_uniqueness(topics_dict=topics_dict, L=args.l)
print(TU)
print(topic_scores)