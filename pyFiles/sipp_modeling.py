import torch
import json 
from transformers import EarlyStoppingCallback
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertConfig
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy.cli
from scipy import spatial
from sentence_transformers import SentenceTransformer
import pandas as pd
import codecs
import itertools
import string
from operator import itemgetter
import gensim
from gensim.models.word2vec import Word2Vec
from gensim import downloader

spacy.cli.download("en_core_web_sm")

# import tensorflow as tf
nltk.download('punkt') 

# variable declaration

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
spacy.cli.download("en_core_web_sm")
pd.set_option('display.max_columns', 500)
w2v = downloader.load('glove-twitter-25')
model_cos = SentenceTransformer('bert-base-nli-mean-tokens')

ref_words =  pd.read_csv('./data/privacy words2.csv')
ref_words = ref_words.set_index('categories')

model0 = BertForSequenceClassification
model1 = BertForSequenceClassification
model2 = BertForSequenceClassification
model3 = BertForSequenceClassification
model4 = BertForSequenceClassification
model5 = BertForSequenceClassification
model6 = BertForSequenceClassification

# function Definitoins

def modeling(): 
    global model0
    global model1
    global model2
    global model3
    global model4
    global model5
    global model6

    model0 = BertForSequenceClassification.from_pretrained("./models/output", num_labels=10)
    model1 = BertForSequenceClassification.from_pretrained("./models/output1", num_labels=2)
    model2 = BertForSequenceClassification.from_pretrained("./models/output2", num_labels=2)
    model3 = BertForSequenceClassification.from_pretrained("./models/output3", num_labels=2)
    model4 = BertForSequenceClassification.from_pretrained("./models/output4", num_labels=2)
    model5 = BertForSequenceClassification.from_pretrained("./models/output5", num_labels=2)
    model6 = spacy.load('en_core_web_sm')

modeling()

def sim_score(d,ref_nouns, ref_verbs,ref_adjectives,ref_pronouns):
    n_count, v_count,a_count, p_count = 5,5,3,1
    d = d.lower()
    d = model6(d)
    tokens = [(i, i.pos_) for i in d if i.text not in string.punctuation]

    nouns = list(set([i[0].text for i in tokens if i[1] == 'NOUN']))
    adjectives = list(set([i[0].text for i in tokens if i[1] == 'ADJ']))
    verbs = list(set([i[0].text for i in tokens if i[1] == 'VERB']))
    pronouns = list(set([i[0].text for i in tokens if i[1] == 'PRON']))

    n_score = distance(nouns,ref_nouns,n_count) / n_count
    v_score = distance(verbs,ref_verbs,v_count) / v_count
    a_score = distance(adjectives,ref_adjectives,a_count) / a_count
    p_score = distance(pronouns,ref_pronouns,p_count) / p_count
    score = (n_score + v_score + a_score +p_score) / 4

    return score

def distance(first_group,second_group,count):
    pairs = list(itertools.product(first_group, second_group))
    try :
        distances = [w2v.similarity(k[0],k[1]) for k in pairs]
        distances.sort(reverse=True)
    except :
        distances = [0] * count
    return sum(distances[:count])



def get_scores(sent):
    cats = ['Data Retention', 'Data Security',
    'First Party Collection/Use', 'Third Party Sharing/Collection',
    'User Access, Edit and Deletion', 'User Choice/Control']
    scores =[]
    hos=0
    for i in cats :
        ref_nouns = ref_words.loc[i]['Nouns'].split(',')
        ref_verbs = ref_words.loc[i]['Verbs'].split(',')
        ref_adjectives = ref_words.loc[i]['Adjectives'].split(',')
        ref_pronouns = ref_words.loc[i]['Pronouns'].split(',')
        scores.append((sim_score(sent,ref_nouns,ref_verbs,ref_adjectives,ref_pronouns),hos))
        hos+=1
    return scores

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def distance_cosine(set1,set2):
    return 1 - spatial.distance.cosine(set1, set2)

# data loading
testing= np.loadtxt("./data/ouput.txt").reshape(6,768)

def load_tokenize_policy():
    with open("./policy_files/amazon_privacy_policy.txt") as f:
        data = f.read()

def sipp_modeling(data):
    tokenize = sent_tokenize(data)
    final_dict = {
    "policy"  : {
        "Data Retention": [],
        "First Party Collection/Use": [],
        "Third Party Sharing/Collection": [],
        "User Access, Edit and Deletion": [],
        "User Choice/Control": [],
        "Data Security": []
        }
    }
    categories = ['Data Retention', 'Data Security', 'First Party Collection/Use', 'Third Party Sharing/Collection','User Access, Edit and Deletion', 'User Choice/Control']
    for sent in (tokenize):
        scores=[0,0,0,0,0,0]
        weight1=0.40
        weight2=0.3
        weight3=0.3
        bert_sent=tokenizer(sent,padding=True,return_tensors='pt')
        out1=softmax(model0(**bert_sent).logits[0].tolist())[1]
        out2=softmax(model1(**bert_sent).logits[0].tolist())[1]
        out3=softmax(model2(**bert_sent).logits[0].tolist())[1]
        out4=softmax(model3(**bert_sent).logits[0].tolist())[1]
        out5=softmax(model4(**bert_sent).logits[0].tolist())[1]
        out6=softmax(model5(**bert_sent).logits[0].tolist())[1]
        tupl_arr=[]
        tupl_arr.append((out1,0))
        tupl_arr.append((out2,1))
        tupl_arr.append((out3,2))
        tupl_arr.append((out4,3))
        tupl_arr.append((out5,4))
        tupl_arr.append((out6,5))
        
        pick_top_two=tupl_arr

        vec=model_cos.encode([sent])[0]
        dis=[]
        for j in range(6):
            dis.append((distance_cosine(testing[j],vec),j))
        top_two=dis

        words_model=get_scores(sent)
        for j in range(6):
            scores[j]+=weight1*pick_top_two[j][0]+weight2*top_two[j][0]+weight3*words_model[j][0]
        
        max_value = max(scores)
        index = scores.index(max_value)
        final_dict["policy"][categories[index]].append(sent)
        #final_dict["policy"][categories[index].append(sent)]
        #print(final_dict)
        #print(type(final_dict))
    return final_dict