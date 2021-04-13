# coding=utf-8
"""
@author: Yantong Lai
@date: 02/02/2020
"""

from transformers import BertTokenizer

import os
import pandas as pd
import argparse


# Define dataset path
WWW2015_sep_path = "../dataset/WWW2015_sep"

# Data file
denmark_file = "denmark.csv"
france_file = "france.csv"
germany_file = "germany.csv"
uk_file = "uk.csv"
us_file = "us.csv"

# Define columns names
column_names = ['text', 'birth year', 'gender', 'rating', 'location']
new_column_names = ['text', 'birth year', 'gender', 'rating', 'location', 'sentiment tokens', 'sentiment values']

# Dictionary file path
AFINN_path = "../dataset/Sentiment_Dictionary/AFINN/AFINN-111.txt"

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', type=str, default='denmark', required=True, help='English reviews of different locations')
args = parser.parse_args()


class AFINN_Dictionary:

    def __init__(self, bert_tokenizer):
        self.bert_tokenizer = bert_tokenizer

    def load(self, dict_path):
        """
        It is a function to load AFINN dictionary.
        :param dict_path: path of the file
        :return: <List> words_list, values_list
        """
        with open(dict_path, 'r') as f:
            lines = f.readlines()
        words_list = []
        values_list = []
        for line in lines:
            words_list.append(line.strip("\n").split("\t")[0])
            values_list.append(int(line.strip("\n").split("\t")[1]))
        return words_list, values_list

    def form_dict(self, words_list, values_list):
        """
        It is a function to form a dictionary, where keys are words_list and values are values_list.
        :param words_list: words_list
        :param values_list: values_list
        :return: <Dict> {words_list: values_list}
        """
        return dict(zip(words_list, values_list))

    def get_sentiment_words(self, senti_dict, text, tokenizer):
        """
        It is a function to get sentiment words.
        """
        text_tokens = tokenizer.tokenize(text)
        senti_tokens = []
        for token in text_tokens:
            if token in senti_dict.keys():
                senti_tokens.append(token)
        return senti_tokens

    def cal_sentiword_values(self, senti_dict, senti_tokens):
        """
        It is a function to calculate values of the senti_tokens.
        """
        senti_values = []
        for token in senti_tokens:
            senti_values.append(senti_dict[token])
        return sum(senti_values)

    def format_tokens(self, senti_tokens_list):
        """
        It is a function to format sentiment tokens list.
        :param senti_tokens_list: <List> sentiment tokens
        :return: formatted sentiment tokens
        """
        format_senti_tokens_list = []
        if isinstance(senti_tokens_list, list):
            for token in senti_tokens_list:
                if not token:
                    format_senti_tokens_list.append("nothing")
                else:
                    format_senti_tokens_list.append(' '.join(token))
        return format_senti_tokens_list

    def add_df_words_values(self, df, senti_dict, tokenizer, col_senti_tokens, col_senti_values):
        """
        It is a function to add senti_tokens and senti_values to the dataframe.
        """
        all_senti_tokens = []
        all_senti_values = []
        for idx in range(len(df)):
            # Get text
            text = df['text'].iloc[idx]
            # Get sentiment tokens, <List>
            senti_tokens = self.get_sentiment_words(senti_dict=senti_dict, text=text, tokenizer=tokenizer)
            all_senti_tokens.append(senti_tokens)
            # Get sentiment values, <List>
            senti_values = self.cal_sentiword_values(senti_dict=senti_dict, senti_tokens=senti_tokens)
            all_senti_values.append(senti_values)

        # Format all_senti_tokens
        format_all_senti_tokens = self.format_tokens(senti_tokens_list=all_senti_tokens)
        assert len(all_senti_tokens) == len(all_senti_values) == len(format_all_senti_tokens) == len(df)
        df.loc[:, col_senti_tokens] = format_all_senti_tokens
        df.loc[:, col_senti_values] = all_senti_values
        return df


def main():
    ###############################
    #    0. Load BERT Tokenizer   #
    ###############################
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    ###############################
    #      1. Load DataFrame      #
    ###############################
    if args.location == 'denmark':
        df = pd.read_csv(os.path.join(WWW2015_sep_path, denmark_file), names=column_names)
    elif args.location == 'france':
        df = pd.read_csv(os.path.join(WWW2015_sep_path, france_file), names=column_names)
    elif args.location == 'germany':
        df = pd.read_csv(os.path.join(WWW2015_sep_path, germany_file), names=column_names)
    elif args.location == 'uk' or args.location == 'united_kingdom':
        df = pd.read_csv(os.path.join(WWW2015_sep_path, uk_file), names=column_names)
    elif args.location == 'us' or args.location == 'united_states':
        df = pd.read_csv(os.path.join(WWW2015_sep_path, us_file), names=column_names)

    ###############################
    #  2. Load AFINN Dictionary   #
    ###############################
    # 1) Create an instance of AFINN_Dictionary
    AFINN_Dict = AFINN_Dictionary(bert_tokenizer=bert_tokenizer)
    print("1) Load AFINN dictionary successfully.")

    # 2) Get words_list and values_list
    words_list, values_list = AFINN_Dict.load(dict_path=AFINN_path)
    print("2) Get words_list and values_list successfully.")

    # 3) Form senti_dict
    senti_dict = AFINN_Dict.form_dict(words_list=words_list, values_list=values_list)
    print("3) Form sentiment dictionary successfully.")

    # 4) Get sentiment words and calculate their values in DataFrame
    df_new = AFINN_Dict.add_df_words_values(df=df, senti_dict=senti_dict, tokenizer=bert_tokenizer,
                                            col_senti_tokens='sentiment tokens',
                                            col_senti_values='sentiment values')
    print("4) Add data to df successfully.")

    # 5) Save dataframes to csv files
    df_new.to_csv(os.path.join(WWW2015_sep_path, args.location + "_sentiment.csv"), index=False, header=False)
    print("5) Save dataframes to csv files successfully.")


if __name__ == '__main__':
    main()
