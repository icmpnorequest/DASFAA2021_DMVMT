# coding=utf-8
"""
It is a Python3 file to split and categorize birth year into groups.
@author: Yantong Lai
@date: 02/15/2020
"""

import pandas as pd
import os
from sklearn.utils import shuffle
import argparse

# Directory path
WWW2015_sep_path = "../dataset/WWW2015_sep"

column_names = ['text', 'birth year', 'gender', 'rating', 'location', 'sentiment tokens', 'sentiment values', 'topic tokens']


# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', type=str, default='denmark', required=True, help='English reviews of different locations')
args = parser.parse_args()


def split_train_dev_test(df, data_path, train_file, dev_file, test_file, train_ratio, dev_ratio, test_ratio):
    """
    It is a function to split total_file to train_file, dev_file and test_file.
    """
    # Shuffle dataframe
    df = shuffle(df)

    df_train = df.iloc[:int(len(df) * train_ratio)]
    df_dev = df.iloc[int(len(df) * train_ratio): int(len(df) * train_ratio) + int(len(df) * dev_ratio)]
    df_test = df.iloc[-int(len(df) * test_ratio):]

    # Save DataFrame to csv file
    df_train.to_csv(os.path.join(data_path, train_file), index=False, header=False)
    df_dev.to_csv(os.path.join(data_path, dev_file), index=False, header=False)
    df_test.to_csv(os.path.join(data_path, test_file), index=False, header=False)


def categorize_age(df):
    """
    It is a function to categorize birth year.

    <=18, label 0;
    (18, 30), label 1;
    [30, 40), label 2;
    [40, 99), label 3.
    """
    df_copy = df

    # Filter
    df_copy = df_copy.drop(df_copy[(df_copy['birth year'] <= 1921) | (df_copy['birth year'] >= 2020)].index)
    birth_year_list = df_copy['birth year'].tolist()

    age_category_list = []
    for year in birth_year_list:
        if (2020 - year) <= 18:
            age_category_list.append(0)
        elif 18 < (2020 - year) < 30:
            age_category_list.append(1)
        elif 30 <= (2020 - year) < 40:
            age_category_list.append(2)
        elif 40 <= (2020 - year) < 99:
            age_category_list.append(3)

    df_copy['birth year'] = age_category_list
    return df_copy


def calculate_age(df):
    """
    It is a function to calculate reviewers' age.
    """
    df_copy = df

    # Filter
    df_copy = df_copy.drop(df_copy[(df_copy['birth year'] <= 1921) | (df_copy['birth year'] >= 2020)].index)
    birth_year_list = df_copy['birth year'].tolist()

    age_list = []
    for year in birth_year_list:
        temp_age = 2020 - year
        age_list.append(temp_age)

    df_copy['birth year'] = age_list
    return df_copy


def main():

    ###############################
    #      1. Load DataFrame      #
    ###############################
    df = pd.read_csv(os.path.join(WWW2015_sep_path, args.location + "_topic.csv"), names=column_names)
    print("1. Read DataFrame successfully.")

    ###############################
    #        2. Categorize        #
    ###############################
    df_cal_age = calculate_age(df=df)
    print("2. Categorize successfully.")

    ###############################
    #          3. Split           #
    ###############################
    split_train_dev_test(df=df_cal_age,
                         data_path=WWW2015_sep_path,
                         train_file=args.location + "_train.csv",
                         dev_file=args.location + "_valid.csv",
                         test_file=args.location + "_test.csv",
                         train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1)
    print("3. Split train, valid and test file successfully.")


if __name__ == '__main__':
    main()
