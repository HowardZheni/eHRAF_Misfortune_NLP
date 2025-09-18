#!/usr/bin/env python
# coding: utf-8

# <font color ="red">NOTE: this code may be archaic and therefore should be used with extreme caution. For more up to date ngram creation, see "N-Grams_MultiLabel.ipynb" which actually should work with single label models but you may need to edit some of the paths and use reduced versions of the codes (which should also be provided).</font>

import pandas as pd
import numpy as np
import json
from datasets import Dataset
import nltk
import re


# Two ways of extracting N-Grams.

# # uncomment based on if you want to do EVENT, CAUSE, or, ACTION
training_label = 'EVENT'
checkpoint = 'checkpoint-264'

# training_label = 'CAUSE'
# checkpoint = 'checkpoint-176'

# training_label = 'ACTION'


# # N-grams from Model Input tokens.

# Here we:
# - Loaded model input Id's
# - Converted them to corresponding words.
# - Added start and stop tokens for Trigrams
# - Got N-grams including start and stop tokens.
# - Got frequency of negative and positive coding
# - Removed N-Grams fewer than 5.
# - Saved data to Excel.
# 

# ## load and clean data

# ### Create tokenizer key/value

# load JSON Tokenizer
with open(f"../HRAF_NLP/HRAF_SingleLabel/HRAF_Model_{training_label}_Demo/{checkpoint}/tokenizer.json") as f:
    tokenizer_base = json.load(f)

# extract only the vocab
tokenizer_base = tokenizer_base['model']['vocab']
# switch key and values
tokenizer = {val: key for key, val in tokenizer_base.items()}


# ### Load model dataset as list

# use one of the two (uncomment):

# from base file
with open(f"../Tokenized_Datasets/tokenized_Hraf_{training_label}.json") as f:
    Hraf = json.load(f)
Hraf


# load optional dataframe.
# df = pd.read_json("../HRAF_NLP/tokenized_Hraf.json", orient='records')


# df.head(3)


# ## Extract N-grams

from collections import defaultdict
from operator import itemgetter
# def N_gram_creator(text, ngram):
#     NGram_list = []
#     for i in range(len(text[:-ngram-1])):
#         NGram_list += [' '.join(text[i:i+ngram])]
#     return NGram_list

# construct list of NGrams
def N_gram_creator(text:list, ngram:int):
    end = text.count("[CLS]") + 1
    temp=zip(*[text[i+(end-ngram):(len(text)-(end-ngram))] for i in range(0,ngram)]) #zip a set of "n" Ngrams. disregard "[sep]" depending on the NGram number.
    ans=[' '.join(ngram) for ngram in temp]
    return ans

# Creates the NGram dictionary
def N_gram_dictionary(passage_dict_list, ngram): #must be a dictionary containing passages as an input
    Ngram_dict = dict()
    # get Ngrams and assign frequencies
    for passage_dict in passage_dict_list:
        # read the tokenized input Ids then add the first and last tokens to the end
        passage_dict['t_words'] = list(itemgetter(*passage_dict['input_ids'])(tokenizer))
        passage_dict['t_words'] = [passage_dict['t_words'][0]] + passage_dict['t_words'] + [passage_dict['t_words'][-1]]
        # Create NGrams and assign frequencies
        for word in N_gram_creator(passage_dict['t_words'], ngram):
            if word not in Ngram_dict.keys():
                Ngram_dict[word] = {'Frequency':0, 'Neg_pred':0,  'Neg_actual':0, "Pos_pred":0, "Pos_actual":0, "Percentage_pred":0, "Percentage_actual":0}
            Ngram_dict[word]['Frequency'] += 1
            # predicted count
            if passage_dict['predicted_label'] == 0:
                Ngram_dict[word]['Neg_pred'] += 1
            elif passage_dict['predicted_label'] == 1:
                Ngram_dict[word]['Pos_pred'] += 1
            else:
                raise ValueError
            # actual count
            if passage_dict['actual_label'] == 0:
                Ngram_dict[word]['Neg_actual'] += 1
            elif passage_dict['actual_label'] == 1:
                Ngram_dict[word]['Pos_actual'] += 1
            else:
                raise ValueError
    # assign percentage (positves/total)
    for Ngram in Ngram_dict.keys():
        Ngram_dict[Ngram]['Percentage_pred'] = Ngram_dict[Ngram]['Pos_pred']/  Ngram_dict[Ngram]['Frequency']
        Ngram_dict[Ngram]['Percentage_actual'] = Ngram_dict[Ngram]['Pos_actual']/  Ngram_dict[Ngram]['Frequency']
    return Ngram_dict


# Save the NGram dictionary
def saveFile(dictionary, fileName, directory, frequency_cutoff=5):
    df2 = pd.DataFrame.from_dict(dictionary, orient='index')
    df2.insert(0, 'N-gram', df2.index)
    df2 = df2.reset_index(drop=True)
    df2 = df2.sort_values(by=['Pos_pred'], ascending=False) 
    # drop all frequencies 5 or less (to shorten the file)
    df2 = df2.loc[df2['Frequency']>=frequency_cutoff]
    df2.to_excel(f'{directory}{fileName}', index=False)
    return fileName + ' Complete'




file_list = [f'_{training_label}_1-grams.xlsx', f'_{training_label}_2-grams.xlsx', f'_{training_label}_3-grams.xlsx']
directory = 'N-Grams_SingleLabel_models/'
# run through each N-gram
for index, file in enumerate(file_list):
    dictionary = N_gram_dictionary(Hraf,index+1 )
    print(saveFile(dictionary=dictionary, fileName=file, directory=directory, frequency_cutoff=5))


# # N-grams From Passage Text.
# 

# here we:
# - Load passages used within the training model (not input ids)
# - Remove all punctuation escept for .!?
# - Lowercase everything.
# - Split passage into tokens.
# - Create N-grams.
# - Remove N-grams that start and end with a stopword.
# - Got frequency of negative and positive coding
# - Removed N-Grams fewer than 5.
# - Saved data to Excel.

# ## Load Data.

# ### Create tokenizer key/value

# load JSON Tokenizer
with open(f"../HRAF_NLP/HRAF_Model_{training_label}_Demo/{checkpoint}/tokenizer.json") as f:
    tokenizer_base = json.load(f)

# extract only the vocab
tokenizer_base = tokenizer_base['model']['vocab']
# switch key and values
tokenizer = {val: key for key, val in tokenizer_base.items()}


# ### Load model dataset as list

# use one of the two (uncomment):

# from base file
with open(f"../Tokenized_Datasets/tokenized_Hraf_{training_label}.json") as f:
    Hraf = json.load(f)
Hraf


# # from dataset
# from datasets import Dataset
# Hraf = Dataset.from_json(f"../Tokenized_Datasets/tokenized_Hraf_{training_label}_Dataset.json")
# Hraf


# ## Extract N-Grams

import string
import copy
import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')

# construct list of NGrams
def N_gram_creator(text, ngram, delete_stopwords='ends'):
    temp=zip(*[text[i:] for i in range(0,ngram)])
    ans=[list(ngram) for ngram in temp]

    # delete all ngrams that contain stopwords
    if delete_stopwords == True:
        ngram_buffer = []
        for ngram in ans:
            if not bool(set(stopwords.words('english')).intersection(ngram)):
                ngram_buffer.append(ngram)
        ans = ngram_buffer
    # delete all ngrams that start or end with stop words (per https://stats.stackexchange.com/questions/570698/should-i-remove-stopwords-before-generating-n-grams)
    elif delete_stopwords == 'ends':
        ngram_buffer = []
        for ngram in ans:
            if (ngram[0] not in set(stopwords.words('english')) and ngram[-1] not in set(stopwords.words('english'))):
                ngram_buffer.append(ngram)
        ans = ngram_buffer

    # turn to string
    ans = [' '.join(ngram) for ngram in ans]
    return ans



# create an N-gram dictionary
def N_gram_dictionary(passage_dict_list, ngram): #must be a dictionary containing passages as an input
    Ngram_dict = dict()
    # get Ngrams and assign frequencies
    for passage_dict in passage_dict_list:
        # return tokenized (and cleaned) passage
        passage_dict['t_words'] = tokenize_words(passage_dict) # return tokenized (and cleaned) passage
        Ngram_passage_count = set() # refresh set for checking if an Ngram has appeared in a passage
        # Create NGrams and assign frequencies
        for word in N_gram_creator(passage_dict['t_words'], ngram):
            # set up the dictionary for that word 
            # (frequency is the number of times the Ngram has appeared in total, _pred refers to the model prediction of negative or positive, _actual refers to the RA label;
            # percentage is positive count divided by frequency, passage_frequency refers to the number of passages the Ngram has appeared where duplicates in a passage are not counted)
            if word not in Ngram_dict.keys():
                Ngram_dict[word] = {'Frequency':0, 'Neg_pred':0,  'Neg_actual':0, "Pos_pred":0, "Pos_actual":0, "Percentage_pred":0, "Percentage_actual":0, "Passage_freq":0, "passage_ID":set()}
            Ngram_dict[word]['Frequency'] += 1
            # assign frequency if the Ngram has not already appeared in the passage.
            if word not in Ngram_passage_count:
                Ngram_passage_count.add(word)
                Ngram_dict[word]['Passage_freq'] += 1
                Ngram_dict[word]['passage_ID'].add(passage_dict['ID'])
            # Get predicted count
            if passage_dict['predicted_label'] == 0:
                Ngram_dict[word]['Neg_pred'] += 1
            elif passage_dict['predicted_label'] == 1:
                Ngram_dict[word]['Pos_pred'] += 1
            else:
                raise ValueError
            # Get actual count
            if passage_dict['actual_label'] == 0:
                Ngram_dict[word]['Neg_actual'] += 1
            elif passage_dict['actual_label'] == 1:
                Ngram_dict[word]['Pos_actual'] += 1
            else:
                raise ValueError
    # assign percentage (positves/total)
    for Ngram in Ngram_dict.keys():
        Ngram_dict[Ngram]['Percentage_pred'] = Ngram_dict[Ngram]['Pos_pred']/  Ngram_dict[Ngram]['Frequency']
        Ngram_dict[Ngram]['Percentage_actual'] = Ngram_dict[Ngram]['Pos_actual']/  Ngram_dict[Ngram]['Frequency']

    return Ngram_dict



# create tokens of the words
def tokenize_words(passage_dict):
    passage_dict['t_words'] = passage_dict['passage'].lower() #lower case everything
    passage_dict['t_words'] = re.sub(r'\S*\.\.\.\S*', ' ', passage_dict['t_words']) # replace elipsis with ' '
    passage_dict['t_words'] = passage_dict['t_words'].split(" ") # get tokens (note that nltk.word_tokenize may be a better option but it messes with non-english text too much)
    # clean up punctuation, remove all but .!? which will become their own token
    word_list = []
    for word in passage_dict['t_words']:
        punctEnd_list = []
        # remove punct from the start of words (some odd punctuation needed to be added).
        safety_count = 0 #include safety count to break in case the file runs too long
        while (len(word) > 0) and (word[0] in string.punctuation or word[0] in '—‘’“”'):
            if word[0] in '?!.':
                word_list += word[0]
            word = word[1:]
            safety_count += 1
            assert safety_count<1000
        # remove punct from the end of words
        while (len(word) > 0) and (word[-1] in string.punctuation or word[0] in '—‘’“”'):
            if word[-1] in '?!.':
                punctEnd_list += word[-1]
            word = word[:-1]
            safety_count += 1
            assert safety_count<1000
        # append if there is something to add.
        if len(word) > 0:
            word_list += [word]
        if len(punctEnd_list) >0:
            word_list += punctEnd_list
    passage_dict['t_words'] = word_list
    return passage_dict['t_words']



# save file (see above function)


file_list = [f'_{training_label}_1-grams_CLEANED.xlsx', f'_{training_label}_2-grams_CLEANED.xlsx', f'_{training_label}_3-grams_CLEANED.xlsx']
directory = 'N-Grams_SingleLabel_models/'
# run through each N-gram (WARNING will take much longer)
for index, file in enumerate(file_list):
    dictionary = N_gram_dictionary(Hraf,index+1 )
    print(saveFile(dictionary=dictionary, fileName=file, directory=directory, frequency_cutoff=5))


# single N-gram
file = f'_{training_label}_8-grams_CLEANED.xlsx'

dictionary = N_gram_dictionary(Hraf,8)
print(saveFile(dictionary, file, frequency_cutoff=2))


# # Explore

# ## Check Token Ids

tokeNum = tokenizer_base['##iology']

df_weird = df[df['input_ids'].apply(lambda x: tokeNum in x)].reset_index(drop=True)
print(tokeNum)
print(df_weird['passage'][0])
print(df_weird['input_ids'][0])


# ### Alternative loading strategies

# various ways to load datafiles. My exploration showed that the "cleanest" version is taking from an unformatted JSON file (which is the smallest file). The second is Lilja's which needs no special packages and just laods by line, third is Dataset.from_json() which is cleaner but requires loading a package, fourth is using ast which is bulkier and has incorrect characters at points

# alternative from Lilja
tokenized_Hraf = [json.loads(line) for line in open("../HRAF_NLP/tokenized_Hraf_Dataset.json", 'r')]
df1 = pd.DataFrame(tokenized_Hraf)


# Load as literal eval (not reccommended)
import ast
dict_list = []
with open("../HRAF_NLP/tokenized_Hraf_Dataset.json", "r") as ins:
    for line in ins:
        data = ast.literal_eval(line)
        dict_list.append(data)
dict_list
ngram_df = pd.DataFrame(dict_list)


dataset = Dataset.from_json("../HRAF_NLP/tokenized_Hraf_Dataset.json")
df3 = pd.DataFrame(dataset)


# comaprison dataframes
x = df1 != df3

for col in x.columns:
    print(sum(x[col]))


# ## Dummy evalutation

import evaluate
precision = evaluate.load('precision')
recall = evaluate.load('recall')
# specif = evaluate.load('specificity')


# import evaluate

# x = np.ones(5)
# x = np.array([1,1,1,1,1,1,1,1,1,0])
# y = np.array([1,1,1,1,1,1,1,1,1,1])
x = np.concatenate((np.ones(900), np.zeros(100)))
y = np.concatenate((np.zeros(15), np.ones(935), np.zeros(50)))

prec = precision.compute(predictions=y, references=x)
rec = recall.compute(predictions=y,references=x)

from sklearn.metrics import f1_score
f1 = f1_score(y_true=x, y_pred=y)
print(f'\n Precision {prec}\nrecall {rec}\n F1 score {f1}')


# ## Dummy check Lilja code
# 

df = pd.read_excel("../RA_Cleaning/Culture_Coding_old.xlsx", header=[0,1], index_col=0)


import time

t_list_list = []
t_df_list = []

loops = 500

for i in range(0, loops):
    t1 = time.time()
    count = 0 
    df_list = df[("ACTION", "No_Info")]
    for action in df_list:
        count += action
    t2 = time.time()
    t_list_list.append(t2-t1)

for i in range(0, loops):
    t1 = time.time()
    count = 0 
    for action in df[("ACTION", "No_Info")]:
        count += action
    t2 = time.time()
    t_df_list.append(t2-t1)


print(f"list: {np.mean(t_list_list)}\ndf:  {np.mean(t_df_list)}")


from collections import defaultdict
from operator import itemgetter

def N_gram_creator(text:list, ngram:int):
    end = text.count("[CLS]") + 1
    temp=zip(*[text[i+(end-ngram):(len(text)-(end-ngram))] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans

def ngram_creator(Hraf, ngram_num):
    unique_Ngrams_set = set()
    Ngrams_list = []
    # get Ngrams and assign frequencies
    for passage_dict in Hraf:
        # read the tokenized input Ids then add the first and last tokens to the end
        passage_dict['t_words'] = list(itemgetter(*passage_dict['input_ids'])(tokenizer))
        passage_dict['t_words'] = [passage_dict['t_words'][0]] + passage_dict['t_words'] + [passage_dict['t_words'][-1]]
        # Create NGrams and assign frequencies
        Ngrams = N_gram_creator(passage_dict['t_words'], ngram_num)
        unique_Ngrams_set.update(Ngrams)
        Ngrams_list += Ngrams
    return  unique_Ngrams_set, Ngrams_list


import time
ngram_df = pd.DataFrame(None, columns = ["ngram", "Frequency"])

# unique_Ngrams_set, Ngrams_list = ngram_creator(Hraf, 2)
# for item in unique_Ngrams_set:
#   freq = Ngrams_list.count(item)
#   ngram_df.loc[len(ngram_df.index)] = [item, freq]
ngram_freq_dict = dict()
count = 0
time1_list = []
time2_list = []
unique_Ngrams_set, Ngrams_list = ngram_creator(Hraf, 3)
for item in unique_Ngrams_set:
    t0 = time.time()
    freq = Ngrams_list.count(item)
    t1 = time.time()
    print(t1 - t0)
    ngram_freq_dict[item] = freq
    t2 = time.time()
    time1_list.append(t1-t0)
    time2_list.append(t2-t1)
    count += 1
    if count%1000 == 0:
        print(f"count: {count}\nNgram_list time: {np.sum(time1_list)}\ndict append time: {np.sum(time2_list)}\n")
        time1_list = []
        time2_list = []
ngram_df = pd.DataFrame(list(ngram_freq_dict.items()), columns=['Ngram', 'Frequency'])
ngram_df


import time



unique_Ngrams_set, Ngrams_list = ngram_creator(Hraf, 2)
Ngrams_list = np.array(Ngrams_list)
unique, freq = np.unique(Ngrams_list, return_counts=True)
ngram_freq_dict = dict(zip(unique, freq))
ngram_df = pd.DataFrame(list(ngram_freq_dict.items()), columns=['Ngram', 'Frequency'])

ngram_df = ngram_df.sort_values(by=['Frequency'], ascending=False) 
# drop all frequencies 5 or less (to shorten the file)
ngram_df = ngram_df.loc[ngram_df['Frequency']>5]
ngram_df.to_excel(f'_{training_label}_2-grams_DUMMY.xlsx', index=False)


ngram_df = pd.DataFrame(None, columns = ["ngram", "Frequency"])

# unique_Ngrams_set, Ngrams_list = ngram_creator(Hraf, 2)
# for item in unique_Ngrams_set:
#   freq = Ngrams_list.count(item)
#   ngram_df.loc[len(ngram_df.index)] = [item, freq]
ngram_freq_dict = dict()
count = 0
time1_list = []
time2_list = []
unique_Ngrams_set, Ngrams_list = ngram_creator(Hraf, 2)
for item in unique_Ngrams_set:
    t0 = time.time()
    freq = Ngrams_list.count(item)
    t1 = time.time()
    ngram_df.loc[len(ngram_df.index)] = [item, freq]
    t2 = time.time()
    time1_list.append(t1-t0)
    time2_list.append(t2-t1)
    count += 1
    if count%1000 == 0:
        print(f"count: {count}\nNgram_list time: {np.sum(time1_list)}\ndict append time: {np.sum(time2_list)}\n")
        time1_list = []
        time2_list = []
# len(unique_Ngrams_set)
ngram_df


# ### Dummy Check Lilja code again.

dictionary = N_gram_dictionary(Hraf,3 )


dictionary['. . .']


matchingHraf_list = []
for passage_dict in Hraf:
    # read the tokenized input Ids then add the first and last tokens to the end
    passage_dict['t_words'] = list(itemgetter(*passage_dict['input_ids'])(tokenizer))
    passage_dict['t_words'] = [passage_dict['t_words'][0]] + passage_dict['t_words'] + [passage_dict['t_words'][-1]]
    if '. . .' in N_gram_creator(passage_dict['t_words'], 3):
        matchingHraf_list.append(passage_dict)


predLabel_list = []
for passage in matchingHraf_list:
    predLabel_list.append(passage['predicted_label'])
sum(predLabel_list)/len(predLabel_list)

