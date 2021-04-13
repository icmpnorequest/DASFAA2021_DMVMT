#!/usr/bin/env bash

# 1. Extract English reviews from raw data file
python3 English_reviews_extractor.py -l denmark
python3 English_reviews_extractor.py -l france
python3 English_reviews_extractor.py -l germany
python3 English_reviews_extractor.py -l uk
python3 English_reviews_extractor.py -l us

# 2. Extract sentiment words
python3 sentiword_extractor.py -l denmark
python3 sentiword_extractor.py -l france
python3 sentiword_extractor.py -l germany
python3 sentiword_extractor.py -l uk
python3 sentiword_extractor.py -l us

# 3. Extract topic words
python3 topic_extractor.py -l denmark
python3 topic_extractor.py -l france
python3 topic_extractor.py -l germany
python3 topic_extractor.py -l uk
python3 topic_extractor.py -l us

# 4. Categorize and split
python3 split_categorize.py -l denmark
python3 split_categorize.py -l france
python3 split_categorize.py -l germany
python3 split_categorize.py -l uk
python3 split_categorize.py -l us