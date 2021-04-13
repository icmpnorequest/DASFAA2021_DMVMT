# coding=utf-8
"""
@author: Yantong Lai
@date: 02/10/2020
"""

from transformers import BertTokenizer

import os
import pandas as pd
import numpy as np
import string
import argparse

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim import corpora, models
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore

from spellchecker import SpellChecker

# Define dataset path
WWW2015_sep_path = "../dataset/WWW2015_sep"

# Sentiment file name
denmark_senti_file = "denmark_sentiment.csv"
france_senti_file = "france_sentiment.csv"
germany_senti_file = "germany_sentiment.csv"
uk_senti_file = "uk_sentiment.csv"
us_senti_file = "us_sentiment.csv"

# Topic file name
denmark_topic_file = "denmark_topic.csv"
france_topic_file = "france_topic.csv"
germany_topic_file = "germany_topic.csv"
uk_topic_file = "uk_topic.csv"
us_topic_file = "us_topic.csv"

# Define columns names
column_names = ['text', 'birth year', 'gender', 'rating', 'location', 'sentiment tokens', 'sentiment values']
new_column_names = ['text', 'birth year', 'gender', 'rating', 'location', 'sentiment tokens', 'sentiment values', 'topic tokens']

# Create the stopwords
stopwords = set(stopwords.words('english'))

# Create the punctuations
punctuations = set(string.punctuation)

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', type=str, default='denmark', required=True, help='Extract topic words of different location')
args = parser.parse_args()


class TFIDF_LDA:

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer

    def get_text(self, filename, col_names):
        """
        It is a function to get text in each row.
        """
        df = pd.read_csv(filename, names=col_names)
        text_list = df['text'].tolist()
        return df, text_list

    def cleaning(self, text, tokenizer, stopwords, puntuations, spell_checker, lemma):
        """
        It is a function to clean the text.
        """
        # 1) Tokenize
        text_tokens = tokenizer.tokenize(text)

        # 2) Correct mis-spelling words
        spell_free = [spell_checker.correction(token) for token in text_tokens]

        # 3) Remove stopwords
        stop_free = [token for token in spell_free if token not in stopwords]

        # 4) Remove punctuations
        punc_free = [token for token in stop_free if token not in puntuations]

        # 5) Lemmatize
        lemma_free = [lemma.lemmatize(token) for token in punc_free]

        print("Text cleaning completed.")
        return [lemma_free]

    def build_dataset(self, clean_text):
        """
        It is a function to build corpus.
        """
        # Create a corpus from a list of texts
        dictionary = Dictionary(clean_text)
        corpus = [dictionary.doc2bow(text) for text in clean_text]
        print("Build dataset successfully.")
        return dictionary, corpus

    def get_topic(self, dictionary, corpus):
        """
        It is a function to build TF-IDF and get topic words
        """
        # 1) Initialize TF-IDF
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]

        # 2) Build LDA model
        topic_list = []
        try:
            lda_model_tfidf = LdaMulticore(corpus_tfidf, num_topics=1, id2word=dictionary, passes=2, workers=4)
            print(lda_model_tfidf.print_topics(-1))
        except ValueError:
            topic_list.append("nothing")

        # Return a <str> object
        return ' '.join(topic_list)


def main():

    ################################
    #       0. Load Instance       #
    ################################
    # 1) Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # 2) Lemmatizer
    lemma = WordNetLemmatizer()

    # 3) Spell Checker
    spell = SpellChecker()
    print("0. Preparation successfully.")

    ################################
    #   1. Load TF-IDF LDA Model   #
    ################################
    # Initialize
    tfidfLDA = TFIDF_LDA(bert_tokenizer=bert_tokenizer)
    print("1. Load TF-IDF LDA model successfully.")

    ################################
    # 2. Complete topic extraction #
    ################################
    print("2. Start Task: {}.".format(args.location))

    # 1) Get train text
    df, text_list = tfidfLDA.get_text(filename=os.path.join(WWW2015_sep_path, args.location + "_sentiment.csv"),
                                      col_names=column_names)

    topics_list = []
    for text in text_list:
        print("No. {}".format(text_list.index(text)))

        # 2) Clean text
        cleaned_text = tfidfLDA.cleaning(text=text,
                                         tokenizer=bert_tokenizer,
                                         stopwords=stopwords,
                                         puntuations=punctuations,
                                         spell_checker=spell,
                                         lemma=lemma)
        # 3) Build dataset
        dictionary, corpus = tfidfLDA.build_dataset(clean_text=cleaned_text)

        # 4) Get topic words
        topic = tfidfLDA.get_topic(dictionary=dictionary, corpus=corpus)
        topics_list.append(topic)

    # 5) Save to new train csv file
    assert len(topics_list) == len(df)
    df.loc[:, 'topic tokens'] = topics_list
    df.to_csv(os.path.join(WWW2015_sep_path, args.location + "_topic.csv"), index=False, header=False)
    print("Save to a new csv file successfully.")


if __name__ == '__main__':
    main()
