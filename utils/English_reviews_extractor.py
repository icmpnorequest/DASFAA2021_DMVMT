# coding=utf-8
"""
@author: Yantong Lai
@date: 02/01/2020
@description: This file aims to extract English reviews with gender and age information in the user profiles.
"""

import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle
import fasttext
import ast
import argparse

# Define file path
WWW2015_raw_path = "../dataset/WWW2015_raw"
WWW2015_sep_path = "../dataset/WWW2015_sep"

# WWW2015 file
Denmark_file = "denmark.auto-adjusted_gender.NUTS-regions.jsonl.tmp"
France_file = "france.auto-adjusted_gender.NUTS-regions.jsonl.tmp"
Germany_file = "germany.auto-adjusted_gender.NUTS-regions.jsonl.tmp"
UK_file = "united_kingdom.auto-adjusted_gender.NUTS-regions.jsonl.tmp"
US_file = "united_states.auto-adjusted_gender.geocoded.jsonl.tmp"

# Define symbol list
SYMBOL_LIST = [",", ".", "-", "/", "[", "]", "?", "<", ">", "{", "}", "|", "\\", ":", ";", "'", "!", "@", "#", "$", "%",
               "_", "(", ")", "\n"]

# fastText language detection model
FASTTEXT_MODEL_PATH = "fastText_model/lid.176.bin"

column_names = ['text', 'birth year', 'gender', 'rating', 'location', 'sentiment tokens', 'sentiment values', 'topic tokens']

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', type=str, default='denmark', required=True, help='English reviews of different locations')
args = parser.parse_args()


class DataHelper:

    def __init__(self):
        pass

    def str_to_dict(self, strdict):
        """
        It is a function change <Str> to <Dict>
        :param strdict: <Str>, like a dict, "{key: value}"
        :return: real <Dict>, {key, value}
        """
        try:
            if isinstance(strdict, str):
                return ast.literal_eval(strdict)
        except Exception as e:
            print(e)
            return None

    def load_file(self, filename):
        """
        It is a function to read file.
        :param filename: filename
        :return: <List> all file lines
        """
        # 0. Load fastText model
        fastText_model = fasttext.load_model(FASTTEXT_MODEL_PATH)

        # 1. Read file
        with open(filename, 'r') as f:
            lines = f.readlines()

        # 2. Change str to dict
        review_list = []
        age_list = []
        gender_list = []
        location_list = []
        rating_list = []

        count = 0
        for line in lines:

            line2dict = self.str_to_dict(line)
            print("No. {}".format(lines.index(line)))

            # 3. Extract review, age, gender and loc
            if 'birth_year' in line2dict.keys() and line2dict['birth_year'] is not None \
                    and 'gender' in line2dict.keys() and line2dict['gender'] is not None \
                    and 'reviews' in line2dict.keys() and line2dict['reviews'] is not None:

                reviews = line2dict['reviews']
                gender = line2dict['gender']
                birth = line2dict['birth_year']
                location = filename.split("/")[-1].split(".")[0]

                # 4. Check if text is written in English
                try:
                    for review in reviews:
                        text = review['text'][0]
                        rating = review['rating']
                        pred = fastText_model.predict(text.strip("\n"))

                        if pred[0][0].replace("__label__", "") == "en" and float(pred[1][0]) >= 0.9:
                            print("text = ", text)
                            print("Add to review_list\n")
                            review_list.append(text)
                            age_list.append(birth)
                            gender_list.append(gender)
                            rating_list.append(rating)
                            location_list.append(location)
                            count += 1
                except IndexError:
                    continue
            else:
                continue
        return review_list, age_list, gender_list, rating_list, location_list
    
    def form_csv(self, review_list, age_list, gender_list, rating_list, location_list, new_path):
        """
        It is a function to save all lists in a csv file.
        :param review_list: review list
        :param age_list: age list
        :param gender_list: gender list
        :param location_list: location list
        :param new_path: new dataset path
        :return: csv file
        """
        # Check the length of the four list objects
        assert len(review_list) == len(age_list) == len(gender_list) == len(rating_list) == len(location_list)

        array = np.array([review_list, age_list, gender_list, rating_list, location_list]).T
        df = pd.DataFrame(data=array)

        # return df.to_csv(os.path.join(new_path, total_filename), index=False, header=False)
        if not os.path.exists(new_path):
            os.mkdir(new_path)

        filename = os.path.join(new_path, location_list[0] + ".csv")
        return df.to_csv(filename, index=False, header=False)


def main():
    
    ####################################
    #    1. Process WWW2015 Data       #
    ####################################
    # 1) Create an instance
    dataHelper = DataHelper()

    # 2) Load file and get English text, users' age, gender ratings and locations
    if args.location == 'denmark':
        review_list, age_list, gender_list, rating_list, location_list = dataHelper.load_file(
            os.path.join(WWW2015_raw_path, Denmark_file))
    elif args.location == 'france':
        review_list, age_list, gender_list, rating_list, location_list = dataHelper.load_file(
            os.path.join(WWW2015_raw_path, France_file))
    elif args.location == 'germany':
        review_list, age_list, gender_list, rating_list, location_list = dataHelper.load_file(
            os.path.join(WWW2015_raw_path, Germany_file))
    elif args.location == 'uk' or args.location == 'united_kingdom':
        review_list, age_list, gender_list, rating_list, location_list = dataHelper.load_file(
            os.path.join(WWW2015_raw_path, UK_file))
    elif args.location == 'us' or args.location == 'united_states':
        review_list, age_list, gender_list, rating_list, location_list = dataHelper.load_file(
            os.path.join(WWW2015_raw_path, US_file))

    # 3) Save to csv file
    # Check if there exists 'WWW2015_sep_path' directory
    if not os.path.exists(WWW2015_sep_path):
        os.mkdir(WWW2015_sep_path)

    # Generate csv file with English reviews, users' age, gender, ratings and locations data
    dataHelper.form_csv(review_list, age_list, gender_list, rating_list, location_list, WWW2015_sep_path)


if __name__ == '__main__':
    main()
