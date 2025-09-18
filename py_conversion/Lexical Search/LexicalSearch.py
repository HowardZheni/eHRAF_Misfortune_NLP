#!/usr/bin/env python
# coding: utf-8

# The following code simply seeks to do one thing, compare accuracy of simple lexical search to the more complicated NLP model <br>
# The code will leverage the idential code in "N-Grams_Multilable.ipynb" to make its classification

# Note, you must have a split

import datasets
import json
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
import os


# # Lexical Search

# ## Set up

# ### Inputs

# CHANGE What model is desired to be used
model = "HRAF_MultiLabel_SubClasses_Kfolds"
# CHANGE Path to model datasets  (you may not need to change this)
path = f"../HRAF_NLP/{model}"
# CHANGE Model dataset Folder
folder = "Datasets"
# CHANGE column name used for passages (typically it is passages but this is to future proof)
passColName = "passage"


# function for loading Json into datasets
def load_json(path):
    f = open(path)
    data = json.load(f)
    dataset = datasets.Dataset.from_dict(data)
    return dataset



# passage_list = [line['passage'] for line in test]
# passage_list


### Dataset partitions, run only one of these code chunks and comment out the rest


# # pre-split training, validation and test datasets
# train = load_json(path=f"{path}/{folder}/train_dataset.json")
# validation = load_json(path=f"{path}/{folder}/validation_dataset.json")
# test = load_json(path=f"{path}/{folder}/test_dataset.json")
# train = datasets.concatenate_datasets([train, validation]) # combine both train and validation
# labels = [label for label in train.features.keys() if label not in ['ID', passColName]]



# pre-split training and test datasets (train contains both train and validation)
train = load_json(path=f"{path}/{folder}/train_dataset.json")
test = load_json(path=f"{path}/{folder}/test_dataset.json")
labels = [label for label in train.features.keys() if label not in ['ID', passColName]]




# # Full dataset that requires splitting after the fact (NOT READY YET)
# with open(f"{path}/tokenized_inputs.json") as f:
#     all = json.load(f)


# ### Ngram functions (run them all)

import string
import copy
import nltk
from nltk.corpus import stopwords
from operator import itemgetter
import re
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


# ## Run Lexical Search via Ngrams

# CHANGE: give a list of N-gram INTEGERS you want to extract (typically [1,2,3] is a good choice )
Ngram_nums = [1,2,3] 
# CHANGE: percentage cutoff in decimal of NGrams with positive rating (i.e. only use Ngrams that are in positively labeled passages x% of the time), You may also put False as a viable input
percentage_cutoff = .85
# CHANGE: choose the frequency an Ngram must appear in general to be allowed to be part of the list ( you do not want to use Ngrams that only appear a single time in the whole dataset)
frequency_cutoff = 5 

df_scores = pd.DataFrame(index = Ngram_nums, columns= [label+"_F1" for label in labels] + ["Micro_F1", "Macro_F1", "Weighted_F1"] )
df_scores.index.name = 'Ngram Number' 

# tokenize the training and test sets first to save on time
trainPass_T = [tokenize_words(passage) for passage in train[passColName]]
testPass_T = [tokenize_words(passage) for passage in test[passColName]]



for Ngram_num in Ngram_nums:
    predictLabels_list = []
    actualLabels_list = []
    for label in labels:
        #extract 'target' Ngrams from training dataset
        passage_list = trainPass_T
        label_list = train[label]
        df_trainTargets = ngram_frequency_creator(passages=passage_list, label_list=label_list, ngram_num=Ngram_num, tokenized_input=True, frequency_cutoff=frequency_cutoff, percentage_cutoff=percentage_cutoff)
        if len(df_trainTargets) == 0:
            print(f'No Ngrams for {Ngram_num} using {label}')
            continue
        targetNgrams_set = set(df_trainTargets['N-gram']) #get only the Ngrams from the result as these will be what we use to predict


        # Predict present or absent based on if target Ngrams appear in the passage (which is also turned into Ngrams)
        testNgrams = [N_gram_creator(tokenizedWords, Ngram_num) for tokenizedWords in testPass_T] # get all the ngrams, technically we could use ngram_frequency_creator() instead since it uses this exact line but this way we skip the superfluous frequency counting!
        predictLabels = [0 if set(PassageNgrams).isdisjoint(targetNgrams_set) else 1 for PassageNgrams in testNgrams] # predict 0 if none of a particular test passage's Ngrams are in the train target Ngrams other wise predict 1
        actualLabels = test[label]

        # Get F1 score for label x Ngram
        df_scores.at[Ngram_num, f"{label}_F1"] = round(f1_score(y_true=actualLabels, y_pred=predictLabels),3)

        # Save predictions and actual for later F1 micro and macro
        predictLabels_list.append(predictLabels)
        actualLabels_list.append(actualLabels)

    #F1 micro and macro score for all the labels
    df_scores.at[Ngram_num, "Micro_F1"] = round(f1_score(y_true=actualLabels_list, y_pred=predictLabels_list, average='micro'),3)
    df_scores.at[Ngram_num, "Macro_F1"]  = round(f1_score(y_true=actualLabels_list, y_pred=predictLabels_list, average='macro'),3)
    df_scores.at[Ngram_num, "Weighted_F1"]  = round(f1_score(y_true=actualLabels_list, y_pred=predictLabels_list, average='weighted'),3)
df_scores = df_scores.astype("Float32")
df_scores


# Show and set up the best model 
largest_idx = df_scores['Micro_F1'].idxmax()
df_score = df_scores.loc[[largest_idx]].copy()
df_score.rename(index={largest_idx: "Lexical search"}, inplace=True)
df_score.index.name = None
print(f"best model (using F1 micro) {largest_idx}")
df_score


# Save prediciton data
from datetime import datetime



# export F1 scores to excel
df_scoresSep = df_score.copy()
# first load train (and maybe add validation)
train = load_json(path=f"{path}/{folder}/train_dataset.json")

if os.path.isfile(path=f"{path}/{folder}/validation_dataset.json"):
    valid = load_json(path=f"{path}/{folder}/validation_dataset.json")
    train = datasets.concatenate_datasets([train, valid])
# add lengths of test and training set
df_scoresSep[["test_length", "train_length"]] = (len(test), len(train))
# add date
df_scoresSep.insert(0, "Date", [datetime.today().date()])
#add optional notes (for Ngrams)
df_scoresSep['Notes'] = f"Ngram {largest_idx}"
# load model_performance.xlsx or else create it
if os.path.isfile(f"{path}/Model_Prediction_Performance.xlsx"):
    df_oldScores = pd.read_excel(f"{path}/Model_Prediction_Performance.xlsx", index_col=0)
    df_oldScores_merged = pd.concat([df_scoresSep, df_oldScores])
    nonDateCols = df_oldScores_merged.columns[df_scoresSep.columns != 'Date']
    if any(df_oldScores_merged.duplicated(subset=nonDateCols)): # don't append the data unless it is new
        print("Duplicated scores found, skipping new addition")
        df_scoresSep = df_oldScores.copy()
    else:
        df_scoresSep = df_oldScores_merged.copy()
        df_scoresSep['Date'] = df_scoresSep['Date'].astype('datetime64[ns]')
        df_scoresSep.to_excel(f"{path}/Model_Prediction_Performance.xlsx")
else:
    df_scoresSep['Date'] = df_scoresSep['Date'].astype('datetime64[ns]')
    df_scoresSep.to_excel(f"{path}/Model_Prediction_Performance.xlsx")

df_scoresSep


# DELETE, this is a test for the zero-division warning seen above.
dummmy_predict = np.array(predictLabels_list)[:,0:10]
print(dummmy_predict)
dummmy_actual = np.array(actualLabels_list)[:,0:10]
print(dummmy_actual)

print(f1_score(y_true=dummmy_actual, y_pred=dummmy_predict, average='macro'))
for predict, actual in zip(dummmy_predict, dummmy_actual):
    recall = recall_score(y_true=actual, y_pred=predict)
    prec = precision_score(y_true=actual, y_pred=predict)
    print(recall, prec)

