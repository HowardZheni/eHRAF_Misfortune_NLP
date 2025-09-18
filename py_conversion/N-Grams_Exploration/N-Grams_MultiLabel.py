#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import json
from datasets import Dataset
import nltk
import re


# CHANGE
# replace checkpoint with intended checkpoint containing tokens Id's (only necessary if you are using nltk tokenizer, otherwise, do not worry about this and skip "Create tokenizer key/value" section)
checkpoint = 'checkpoint-700'
# replace model with the name of the model wanted to use
model = "ThreeLargeClasses"
# Edit path as you see fit to reach location where all files associated with the model's output are
directory = f"../HRAF_NLP/HRAF_MultiLabel_{model}"


# # N-grams From Passage Text.
# 

# here we:
# - Load passages used within the training model (not input ids)
# - Remove all punctuation escept for .!?
# - Lowercase everything.
# - Split passage into tokens.
# - Optionally delete all stopwords 
# - Create N-grams.
# - Optionally remove N-grams with stopwords (either at all or only at the ends) if stopwords have not already been removed
# - Got frequency of negative and positive coding
# - Removed N-Grams fewer than 5.
# - Saved data to Excel.

# ## Set up (run all)

# ### Create tokenizer key/value (Only relevant if you do not want to use my tokenizer and prefer NLTK tokenizer instead)

# load JSON Tokenizer (note, you may need to change this if you do not follow the naming convention I am using. ultimately, this code should work with single label ngrams!)
with open(f"{directory}/HRAF_Model_MultiLabel_{model}/{checkpoint}/tokenizer.json") as f:
    tokenizer_base = json.load(f)

# extract only the vocab
tokenizer_base = tokenizer_base['model']['vocab']
# switch key and values
tokenizer = {val: key for key, val in tokenizer_base.items()}


# ### Load model dataset as list

# Load Tokenized passages and inputs
with open(f"{directory}/Datasets/tokenized_inputs.json") as f:
    Hraf = json.load(f)


# ### Ngram Functions

import string
import copy
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import time
# nltk.download('stopwords')



def N_gram_creator(text:list, ngram:int, delete_stopwords:bool=False, use_end_tokens:bool=False, fuse_ngrams:bool=True):

    # whether or not to include end tokens inside the datasets
    if use_end_tokens == True:
        start = text.count("[CLS]") + 1
        end = text.count("[SEP]") + 1
        assert start == end, "Unequal number of start and end tokens"
        if start < ngram:
            text = ['[CLS]'] * (ngram- start) + text + ['[SEP]'] * (ngram- end)
            start = text.count("[CLS]") + 1
            end = text.count("[SEP]") + 1
        temp=zip(*[text[i+(start-ngram):(len(text)-(end-ngram))] for i in range(0,ngram)]) #zip a set of "n" Ngrams. disregard "[sep]" depending on the NGram number.
    else:
        # create Ngram by zipping
        temp=zip(*[text[i:] for i in range(0,ngram)])
    ans=[list(ngram) for ngram in temp]

    # delete all ngrams that contain stopwords (after the fact creation of Ngrams)
    if delete_stopwords == True:
        ngram_buffer = []
        cached_stopwords = set(stopwords.words('english'))
        for ngram in ans:
            if not bool(cached_stopwords.intersection(ngram)):
                ngram_buffer.append(ngram)
        ans = ngram_buffer
    # delete all ngrams that start or end with stop words (per https://stats.stackexchange.com/questions/570698/should-i-remove-stopwords-before-generating-n-grams)
    elif delete_stopwords == 'ends':
        ngram_buffer = []
        for ngram in ans:
            if (ngram[0] not in set(stopwords.words('english')) and ngram[-1] not in set(stopwords.words('english'))):
                ngram_buffer.append(ngram)
        ans = ngram_buffer

    # Optionally turn to string
    if fuse_ngrams:
        ans = [' '.join(ngram) for ngram in ans]
    return ans

def N_gram_dictionary(passage_dict_list_copy, ngram_num:int, label:str, passagefreq:bool=True, use_end_tokens:bool=False, use_tokenized:bool=False, tokenizer=False, text_name:str='passage', id_name:str='ID', predLabel_name:str='pred_labels', actualLabel_name:str='actual_labels'): #must be a dictionary containing passages as an input
    """
    Create a dataframe composed of specified Ngram and label

    Parameters:
    ----------
    passage_dict_list_copy : list of dictionaries containing text for Ngrams, predicted labels, and actual labels. Typically derived from machine learning model
    ngram_num : int, ngram number
    label : string, label used for Ngram
    passagefreq : boolean, should it count the frequency of passages occuring?
    use_end_tokens : boolean, should we include end tokens like [CLS] as ngrams?
    use_tokenized : boolean, is the dataframe already tokenized
    tokenizer : NLP tokenizer, you must supply what that tokenizer is!
    text_name : string, column name of text used for Ngrams
    id_name : string, column name of IDs
    predLabel_name : string, column name of predicted label
    actualLabel_name : string, column name of actual label

    Returns:
    -------
    Ngram_dict : list of dictionaries containing Ngrams and their frequencies
    """

    if use_tokenized is True:
        assert tokenizer is not False, "Must supply a tokenizer"

    Ngram_dict = dict()
    passage_dict_list = copy.deepcopy(passage_dict_list_copy)
    # get Ngrams and assign frequencies
    for passage_dict in passage_dict_list:
        # return tokenized (and cleaned) passage
        if use_tokenized is True:
            passage_dict['t_words'] = list(itemgetter(*passage_dict['input_ids'])(tokenizer))
            passage_dict['t_words'] = [passage_dict['t_words'][0]] + passage_dict['t_words'] + [passage_dict['t_words'][-1]]
        else:
            passage_dict['t_words'] = tokenize_words(passage_dict[text_name])

        Ngram_passage_count = set() # refresh set for checking if an Ngram has appeared in a passage
        # Create NGrams and assign frequencies
        for word in N_gram_creator(passage_dict['t_words'], ngram_num, use_end_tokens):
            # set up the dictionary for that word 
            # (frequency is the number of times the Ngram has appeared in total, _pred refers to the model prediction of negative or positive, _actual refers to the RA label;
            # percentage is positive count divided by frequency, passage_frequency refers to the number of passages the Ngram has appeared where duplicates in a passage are not counted)
            if word not in Ngram_dict.keys():
                if passagefreq is True: # count passage frequency
                    Ngram_dict[word] = {'Frequency':0, 'Neg_pred':0,  'Neg_actual':0, "Pos_pred":0, "Pos_actual":0, "Percentage_pred":0, "Percentage_actual":0, "Passage_freq":0, "passage_ID":set()}
                else: # don't count passage frequency
                    Ngram_dict[word] = {'Frequency':0, 'Neg_pred':0,  'Neg_actual':0, "Pos_pred":0, "Pos_actual":0, "Percentage_pred":0, "Percentage_actual":0} 
            Ngram_dict[word]['Frequency'] += 1

            # assign frequency if the Ngram has not already appeared in the passage.
            if passagefreq is True and word not in Ngram_passage_count:
                Ngram_passage_count.add(word)
                Ngram_dict[word]['Passage_freq'] += 1
                Ngram_dict[word]['passage_ID'].add(passage_dict[id_name])


            # Get predicted count
            if passage_dict[predLabel_name][label] == 0:
                Ngram_dict[word]['Neg_pred'] += 1
            elif passage_dict[predLabel_name][label] == 1:
                Ngram_dict[word]['Pos_pred'] += 1
            else:
                raise ValueError

            # Get actual count
            if passage_dict[actualLabel_name][label] == 0:
                Ngram_dict[word]['Neg_actual'] += 1
            elif passage_dict[actualLabel_name][label] == 1:
                Ngram_dict[word]['Pos_actual'] += 1
            else:
                raise ValueError
    # assign percentage (positves/total)
    for Ngram in Ngram_dict.keys():
        Ngram_dict[Ngram]['Percentage_pred'] = Ngram_dict[Ngram]['Pos_pred']/  Ngram_dict[Ngram]['Frequency']
        Ngram_dict[Ngram]['Percentage_actual'] = Ngram_dict[Ngram]['Pos_actual']/  Ngram_dict[Ngram]['Frequency']

    return Ngram_dict


def tokenize_words(passage:str, removeStopWords=True):
    passage = passage.lower() #lower case everything
    passage = re.sub(r'\.\.\.', ' ', passage) # replace elipsis with ' '
    passage_t = passage.split(" ") # get tokens (note that nltk.word_tokenize may be a better option but it messes with non-english text too much)


    # clean up punctuation, remove all but .!? which will become their own token
    word_list = []
    for word in passage_t:
        # optional remove stopwords
        punctEnd_list = []
        # remove punct from the start of words (some odd punctuation needed to be added).
        safety_count = 0 #include safety count to break in case the file runs too long
        punct = string.punctuation+'—‘’“”'
        while (len(word) > 0) and word[0] in punct:
            if word[0] in '?!.':
                word_list += word[0]
            word = word[1:]
            safety_count += 1
            assert safety_count<1000
        # remove punct from the end of words
        while (len(word) > 0) and word[-1] in punct:
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
    if removeStopWords is True: 
        cached_stopwords = set(stopwords.words('english'))
        word_list = [word for word in word_list if word not in cached_stopwords]
    passage_t = word_list
    return passage_t

# Save the NGram dictionary
def saveFile(dictionary, fileName, folder, frequency_cutoff=5):
    make_dir(folder)
    df2 = pd.DataFrame.from_dict(dictionary, orient='index')
    df2.insert(0, 'N-gram', df2.index)
    df2 = df2.reset_index(drop=True)
    df2 = df2.sort_values(by=['Pos_pred'], ascending=False) 
    # drop all frequencies 5 or less (to shorten the file)
    df2 = df2.loc[df2['Frequency']>=frequency_cutoff]

    df2.to_excel(f'{folder}/{fileName}', index=False)
    return fileName + ' Complete'

# made directory
def make_dir(path):
    import os
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
    # Create a new directory because it does not exist
        os.makedirs(path)


# only needs "actual labels" and text, spits out pandas dataframe for frequency
def ngram_frequency_creator(passages:list, label_list:list, ngram_num:int, tokenized_input:bool = False, frequency_cutoff:int=5, percentage_cutoff:float=False):
    assert isinstance(passages, list), "ERROR passage Input must be a list of passages"
    # get tokens and then get ngrams from those tokens
    if tokenized_input == False: # does your input need to be tokenized?
        assert ~isinstance(passages[0], list), "ERROR Your Passage list input appears to be a list of lists, are you sure you didn't mean to select tokenized_input = True?"
        tokenizedWords_list = [tokenize_words(passage) for passage in passages]
    else:
        tokenizedWords_list = passages
    NgramTokens_list = [N_gram_creator(tokenizedWords, ngram_num) for tokenizedWords in tokenizedWords_list]
    # get dictionary of Ngram frequency and actual label
    Ngram_dict = dict()
    for index, NgramTokens in enumerate(NgramTokens_list):
        for word in NgramTokens:
            if word not in Ngram_dict.keys():
                Ngram_dict[word] = {'Frequency':0, 'Neg_actual':0, "Pos_actual":0, "Percentage_actual":0} 
            Ngram_dict[word]['Frequency'] += 1

            # Get actual count
            if label_list[index] == 0:
                Ngram_dict[word]['Neg_actual'] += 1
            elif label_list[index] == 1:
                Ngram_dict[word]['Pos_actual'] += 1
            else:
                raise ValueError
    # assign percentage (positves/total)
    for Ngram in Ngram_dict.keys():
        Ngram_dict[Ngram]['Percentage_actual'] = Ngram_dict[Ngram]['Pos_actual']/  Ngram_dict[Ngram]['Frequency']

    # create dataframe and remove those below cutoff
    df2 = pd.DataFrame.from_dict(Ngram_dict, orient='index')
    df2.insert(0, 'N-gram', df2.index)
    df2 = df2.reset_index(drop=True)

    # drop all frequencies that are too small
    df2 = df2.loc[df2['Frequency']>=frequency_cutoff]

    # delete those with too small frequencies
    if percentage_cutoff:
        df2 = df2.loc[df2['Percentage_actual']>=percentage_cutoff]

    return df2


# ## Call functions

# ### Get Ngram dictionaries 

# This code creates a series of Ngrams from a range of inputs such as NTLK tokenized passages, tokenized passages using this code, and raw passages and corresponding labels <br>
# Inputs to N_gram_dictionary():

#  
#     passage_dict_list_copy : list of dictionaries containing text for Ngrams, predicted labels, and actual labels. Typically derived from machine learning model
#     ngram_num : int, ngram number
#     label : string, label used for Ngram
#     passagefreq : boolean, should it count the frequency of passages occuring?
#     use_end_tokens : boolean, should we include end tokens like [CLS] as ngrams?
#     use_tokenized : boolean, is the dataframe already tokenized
#     tokenizer : NLP tokenizer, you must supply what that tokenizer is (hint, you might use the toknizer defined at the very top)
#     text_name : string, column name of text used for Ngrams
#     id_name : string, column name of IDs
#     predLabel_name : string, column name of predicted label
#     actualLabel_name : string, column name of actual label

# CHANGE: give a list of N-gram INTEGERS you want to extract
Ngram_nums = [1,2,3] 
# CHANGE: include the suffix at the end of each of these files if you want to keep multiple versions of these files (may be blank)
suffix = ''

folder = f'{directory}/Ngrams' # Doesn't need changing (will create a folder automatically)


for label in Hraf[0]['pred_labels'].keys():
    # run through each N-gram
    for Ngram_num in Ngram_nums:
        file = f'_{label}_{Ngram_num}-grams{suffix}.xlsx'
        dictionary = N_gram_dictionary(Hraf, Ngram_num, label, passagefreq=True) # get NGram dictionary
        print(saveFile(dictionary=dictionary, fileName=file, folder=folder, frequency_cutoff=5)) # Save NGram dictionary


# ### Get simplified Ngrams based on Passage input and label inputs

# This code is a simplified version of the one above as it cannot accept model tokenized inputs, and has far fewer options. However, its simplicity is to its benefit. This code will output Ngrams with removed stopwords 4 times as fast as the above code and does not require a prexisting model to work. All you need is a list of passages and a list of labels (e.g. 1's and 0's). NOTE if you are wondering whether to use this section or the above section, both will work just fine but the above dictionary code gives more info for the Ngrams)

# I believe the intention was to have an extremely toned down version (also faster) of the Ngram creater which only needs a list of the inputs
passage_list = [line['passage'] for line in Hraf]
label_list = [line['actual_labels']['EVENT'] for line in Hraf]


df3 = ngram_frequency_creator(passages=passage_list, label_list=label_list, ngram_num=3, frequency_cutoff=5, percentage_cutoff=False)
df3.sort_values(by="Frequency", ascending=False)


# Get each Ngram per label based on a list of passages and a list of lables

### This cell replicates the above ngram dictionary cell.
# CHANGE: give a list of N-gram INTEGERS you want to extract
Ngram_nums = [1,2,3] 
# CHANGE: include the suffix at the end of each of these files if you want to keep multiple versions of these files (may be blank)
suffix = '_alt'


folder = f'{directory}/Ngrams' # Doesn't need changing (will create a folder automatically)

labels = Hraf[0]['pred_labels'].keys() # you may change this to whatever but if you do not have a list of lists [[label1], [label2], [label3]]  (i.e. you only have a single label, it is reccommended to still make that label inside a second list: [[label]] )

# Get list of passages (then tokenize them to make it faster)
passage_list = [line['passage'] for line in Hraf]
passage_list_t = [tokenize_words(passage) for passage in passage_list]

# Run through each label N-gram pair (Note, the more combinations you have, the more important it is to tokenize your data first)
for label in labels:
    label_list = [line['actual_labels'][label] for line in Hraf]
    # run through each N-gram (WARNING will take much longer than previous N-gram
    for Ngram_num in Ngram_nums:
        file = f'_{label}_{Ngram_num}-grams{suffix}.xlsx'
        df3 = ngram_frequency_creator(passages=passage_list_t, label_list=label_list, ngram_num=Ngram_num, tokenized_input=True, frequency_cutoff=5, percentage_cutoff=False)
        df3 = df3.sort_values(by="Frequency", ascending=False)
        df3.to_excel(f'{folder}/{file}', index=False)
        print(file, "Complete")

