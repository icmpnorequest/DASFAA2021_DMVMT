# Data Pre-processing Description


## 1. Files Descriptions

### 1) English_reviews_extractor.py

- It aims to extract English reviews from the raw five datasets: 
  - Denmark
  - France
  - Germany
  - UK
  - US

### 2) sentiword_extractor.py

- It aims to extract sentiment words according to AFINN, a sentiment directory, which is contained and cited under the `dataset` directory.

### 3) topic_extractor.py

- It aims to extract topics with LDA, a traditional topic model.

### 4) process.sh

- File 1) - 3) aims to pre-process the raw datasets in three aspects, so we build a pipeline torun 1) - 3) files together.


## 2. How to Pre-process the Datasets

### 2.1 Download Language Identification Models from fastText

Step 1: Download `lid.176.bin` from [link](https://fasttext.cc/docs/en/language-identification.html).

Step 2: Make directory `fastText_model` under the `utils` and copy `lid.176.bin` into `fastText_model`, like following:

```
utils
├── English_reviews_extractor.py
├── README.md
├── fasttext_model
│   └── lid.176.bin
├── process.sh
├── sentiword_extractor.py
├── split_categorize.py
└── topic_extractor.py
```


### 2.2 Run process.sh

```
# Step 1: cd utils
$ cd utils

# Step 2: Add permission with process.sh
$ chmod +x ./process.sh

# Step 3: Run process.sh
$ ./process.sh
```

