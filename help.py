import os, re, sys, time, codecs
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from inspect import getargspec

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from string import punctuation


def text_to_wordlist(text, remove_stopwords=False, stem_words=False, remove_punc=False, keep_period=True):
    # Clean the text, with the option to remove stopwords and to stem words.

    # Convert words to lower case and split them
    text = text.lower().split()

    # [Optional] remove stop words
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]

    text = " ".join(text)

    # Clean the text
    if remove_punc:
        if keep_period:
            text = re.sub(r"[^A-Za-z0-9@\.\!\?]", " ", text)
        else:
            text = re.sub(r"[^A-Za-z0-9@]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " / ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " U.S. ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    # [Optional] shorten words to their stems
    if stem_words:
        text = text.split()
        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]
        text = " ".join(stemmed_words)

    # Return a list of words
    return (text.strip())


def load_dataset(data_file, name, first=100000000, remove_stopwords=False, stem_words=False, remove_punc=False,
                 keep_period=True):
    datasets = {"news": [], "questions": [], "answers": []}
    entitiesDataset = []
    count = 0

    for subdir, dirs, files in os.walk(data_file):
        for file in files:
            filepath = os.path.join(subdir, file)
            if filepath.endswith(".question"):
                with codecs.open(filepath, encoding='utf-8') as f:
                    html, news, question, answer, entities = f.read().split("\n\n")
                    datasets["news"].append(
                        text_to_wordlist(news, remove_stopwords, stem_words, remove_punc, keep_period))
                    datasets["questions"].append(
                        text_to_wordlist(question, remove_stopwords, stem_words, remove_punc, keep_period))
                    datasets["answers"].append(answer)
                    entitiesDataset.append((news, question, answer, entities))

                    count += 1
                    if count % 1000 == 0:
                        print("Finished {} questions in {}".format(count, name))
                    if count > first:
                        break
        if count > first:
            break

    return datasets, entitiesDataset


def X_y_gen(index, news, questions, answers):
    """
    1. frequency: entity frequencies in the news
    2. first index position: 80% of entities show up in the 1/3 of the whole contexts
    3. distance: the minimum distance between each entities in current news set each enti- ties or clues in the question data
    4. bigram exact match: whether appears in [-4, 4]
    5. within top 5 similarity: whether appers in top 5

    output: [frequency, first_pos, distance, n_gram_match, similar] [0] for all entities in a news
    """

    frequency = 0
    first_pos = 0  # percentage
    distance = 0
    n_gram_match = 0  # 0 or 1
    similar = 0  # 0 or 1 # 24.4% in top 5 on average

    news_raw, questions_raw, answer = news[index], questions[index], answers[index]

    news_tokens = [k for j in [i.split() for i in news_raw.split(" . ")] for k in j]
    news_sents = [j for j in [i.split() for i in news_raw.split(" . ")] if len(j) > 5] + [questions_raw.split()]

    questions = questions_raw.split()
    clues = [i for i in questions if i != '@placehold']
    q_key = [i.strip() for i in re.search(r"(\S+ ){,2}@placehold( \S+){,2}", questions_raw)[0].split("@placehold")]

    entities_per_news = sorted(
        list(set([w for w in news_tokens if w.startswith('@entity')])))  # all unique entites in this passage

    X_per_news = []
    y_per_news = []
    for e in entities_per_news:

        # 1. frequency
        counter = news_tokens.count(e)
        frequency = counter

        # 2. first index location
        loc = news_tokens.index(e)
        pos = loc / len(news_tokens)
        first_pos = pos

        # 3. distance
        tep = float('Inf')
        for c in clues:
            try:
                key_c = news_tokens.index(c)
                if abs(loc - key_c) < tep:
                    tep = abs(loc - key_c)
            except:
                tep = 0
        distance = tep

        # 4. bigram exact match
        ent_pos = [i for i in range(len(news_tokens)) if news_tokens[i] == e]
        for i in ent_pos:
            if i < 4:
                bigram = [" ".join(news_tokens[:i]), " ".join(news_tokens[i + 1:i + 5])]
            elif i > len(news_tokens) - 4:
                bigram = [" ".join(news_tokens[i - 4:i]), " ".join(news_tokens[i + 1:])]
            else:
                bigram = [" ".join(news_tokens[i - 4:i]), " ".join(news_tokens[i + 1:i + 5])]
            for q in q_key:
                for b in bigram:
                    if q in b and q != "" and b != "":
                        n_gram_match = 1
                        break

        # append to X, y
        X_per_news.append([frequency, first_pos, distance, n_gram_match])
        if e == answer:
            y_per_news.append(1)
        else:
            y_per_news.append(0)

    return X_per_news, y_per_news, entities_per_news
