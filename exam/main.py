import csv
import string
from itertools import islice
import spacy
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from enum import Enum
import Stemmer
import pickle

nlp = spacy.load('lt_core_news_lg')
embedder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
processed = []
stemmer = Stemmer.Stemmer('lithuanian')

synonyms = []
with open('exam/wordnet.txt') as f:
    lines = f.readlines()
    for line in lines:
        words = line.split('␞')[0].split(';')
        if len(words) > 1:
            current_synonyms = []
            if len(words[0].split()) == 3:
                current_synonyms.append(words[0].split()[2].strip())
            for word in words[1:]:
                if len(word.split()) == 1:
                    current_synonyms.append(word.split('\t')[0].strip())
            if len(current_synonyms) > 1:
                synonyms.append(current_synonyms)

def lemmatize(text):
    doc = nlp(text)
    text_lemmatized_list = []

    for token in doc:
        if token.lemma_ != "-PRON-": 
            text_lemmatized_list.append(token.lemma_)
        else: 
            text_lemmatized_list.append(token)
    text_lemmatized = ' '.join(text_lemmatized_list)

    return text_lemmatized

def lemmatize_list(texts):
    lemmatized = []
    for text in texts:
        lemmatized.append(lemmatize(text))
    return lemmatized

def stem_string(text):
    return ' '.join(stemmer.stemWords(text.split()))

def stem_list(texts):
    stemmed = []
    for text in texts:
        stemmed.append(stem_string(text))
    return stemmed

with open('exam/processed.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        text = row[1]
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        without_short_words = ""
        for word in text.split():
            if not (word.isdigit() or len(word) < 3):
                without_short_words += word + " "
        processed.append([row[0], lemmatize(without_short_words)])
        # processed.append([row[0], stem_string(without_short_words)])
        # processed.append([row[0], without_short_words])
del processed[0]

# reading the base bigrams, converting them to arrays and cleaning up
pos_bigrams = pd.DataFrame(pd.read_excel("exam/T_N_2gramos.xlsx", engine='openpyxl')).to_numpy()
unclean_neg_bigrams = pd.DataFrame(pd.read_excel("exam/T_N_2gramos.xlsx", engine='openpyxl', sheet_name="Neigiamos")).to_numpy()
neg_bigrams = unclean_neg_bigrams[:, :2]


pos_bi = []
pos_bi_wordnet = []
for bigram in pos_bigrams:
    stringed_bigram = bigram[0] + " " + bigram[1]
    pos_bi.append(stringed_bigram)

    for words in synonyms:
        if bigram[0] in words:
            for syn in words:
                pos_bi_wordnet.append(syn + " " + bigram[1])
        if bigram[1] in words:
            for syn in words:
                pos_bi_wordnet.append(bigram[0] + " " + syn)
    if stringed_bigram not in pos_bi_wordnet:
        pos_bi_wordnet.append(stringed_bigram)

neg_bi = []
neg_bi_wordnet = []
for bigram in neg_bigrams:
    stringed_bigram = bigram[0] + " " + bigram[1]
    neg_bi.append(stringed_bigram)

    for words in synonyms:
        if bigram[0] in words:
            for syn in words:
                neg_bi_wordnet.append(syn + " " + bigram[1])
        if bigram[1] in words:
            for syn in words:
                neg_bi_wordnet.append(bigram[0] + " " + syn)
    if stringed_bigram not in neg_bi_wordnet:
        neg_bi_wordnet.append(stringed_bigram)

print(len(neg_bi), len(set(neg_bi_wordnet)), len(pos_bi), len(set(pos_bi_wordnet)))

filtered_bigrams = []
with open('exam/bigrams_filtered.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        text = row[0]
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        if len(min(text.split(), key=len)) >= 3:
            filtered_bigrams.append(text)

print("Number of bigrams: ", len(filtered_bigrams))

def balance_dataset(dataset, count):
    balanced = []
    pos_count = 0
    neg_count = 0
    neu_count = 0
    for data in dataset:
        if data[0] == "POS" and pos_count < count:
            pos_count += 1
            balanced.append(data)
        elif data[0] == "NEU" and neu_count < count:
            neu_count += 1
            balanced.append(data)
        elif data[0] == "NEG" and neg_count < count:
            neg_count += 1
            balanced.append(data)
        if pos_count == count and neg_count == count and neu_count == count:
            break
    return balanced


# corpus_embeddings = embedder.encode(filtered_bigrams, convert_to_tensor=True)
# # Store sentences & embeddings on disc
# with open('embeddings.pkl', "wb") as fOut:
#     pickle.dump({'embeddings': corpus_embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

#Load sentences & embeddings from disc
with open('embeddings.pkl', "rb") as fIn:
    stored_data = pickle.load(fIn)
    corpus_embeddings = stored_data['embeddings']

def expand_lexicon(base):
    expanded = base.copy()
    query_embeddings = embedder.encode(base, convert_to_tensor=True)
    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=5)
    for similars in hits:
        for entry in similars:
            if entry['score'] > 0.8:
                expanded.append(filtered_bigrams[entry["corpus_id"]])
    return expanded

expanded_pos = expand_lexicon(pos_bi_wordnet)
expanded_neg = expand_lexicon(neg_bi_wordnet)

stemmed_pos = stem_list(pos_bi)
stemmed_neg = stem_list(neg_bi)

s_e_pos = stem_list(expanded_pos)
s_e_neg = stem_list(expanded_neg)

# lemm_pos = lemmatize_list(pos_bi)
# lemm_neg = lemmatize_list(neg_bi)

# l_e_pos = lemmatize_list(expanded_pos)
# l_e_neg = lemmatize_list(expanded_neg)

def remove_reverses(bigrams):
    fixed = []
    for bigram in bigrams:
        words = bigram.split()
        if words[1] + " " + words[0] not in bigrams:
            fixed.append(bigram)
    return fixed

fixed_pos = remove_reverses(s_e_pos)
fixed_neg = remove_reverses(s_e_neg)

print("Final lengths:", len(set(fixed_pos)), len(set(fixed_neg)))

def bag_of_bigrams(text):
    bag = []
    words = text.split()
    for i in range(len(words) - 1):
        bag.append(words[i] + " " + words[i+1])
        bag.append(words[i+1] + " " + words[i])
    return bag


def evaluate(pos_bigrams, neg_bigrams, texts):
    conf_matrix = np.zeros((3, 3))
    true_sentiment = {"POS" : 0, "NEU" : 1, "NEG" : 2}
    for text in texts:
        pos_count = 0
        neg_count = 0
        bag = bag_of_bigrams(text[1])
        for bigram in bag:
            if bigram in pos_bigrams:
                pos_count += 1
            elif bigram in neg_bigrams:
                neg_count += 1
        # relative proportional difference
        if pos_count + neg_count == 0:
            score = 0
        else:
            score = (pos_count - neg_count) / (pos_count + neg_count)

        if score > 0:
            pred_sent = 0
        elif score < 0:
            pred_sent = 2
        else:
            pred_sent = 1
        conf_matrix[pred_sent][true_sentiment[text[0]]] += 1
    correct = conf_matrix[0, 0] + conf_matrix[1, 1] + conf_matrix[2, 2]
    count = np.sum(conf_matrix)
    acc = correct / count
    return conf_matrix, acc


result, acc = evaluate(stemmed_pos, stemmed_neg, processed)
print(result)
print(acc)

result, acc = evaluate(fixed_pos, fixed_neg, processed)
print(result)
print(acc)

result, acc = evaluate(stemmed_pos, stemmed_neg, balance_dataset(processed, 1500))
print(result)
print(acc)

result, acc = evaluate(fixed_pos, fixed_neg, balance_dataset(processed, 1500))
print(result)
print(acc)
