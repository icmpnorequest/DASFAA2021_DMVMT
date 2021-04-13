# DASFAA2021_Neural Demographic Prediction in Social Media with Deep Multi-View Multi-Task Learning

@author: Yantong Lai, Yijun Su, Cong Xue, Daren Zha


## 1. Code Description

### 1) dataset

Under the `dataset` directory, it mainly contains three parts:

- `WWW2015_raw` contains raw datasets from datasets proposed by Hovy et al. [1];
- `WWW2015_sep` saves files after pre-processing;
- `Sentiment_Dictionary` provides sentiment directory, AFINN [2].

### 2) model

- DMVMT.py: Our proposed model

### 3) utils

- English_reviews_extractor.py: Extract Enligsh reviews with gender and age information from raw datasets
- sentiword_extractor.py: Extract sentiment words
- Topic_extractor.py: Extract topics
- Split_categorize.py: Split processed datasets into train, valid and test files and categorize birth year into four groups
- process.sh: Pipeline to process datasets 


## 2. How to Run Our DMVMT?

### Step 1: Download raw datasets

Please follow the guide under dataset/WWW2015_raw to download the raw datasets.

### Step 2: Pre-process raw datasets

Please follow the guide under utils/ to run `process.sh`

It would take some time, for both uk and us datasets are large.

### Step 3: Run model
```
$ cd model/
$ python3 DMVMT.py -l denmark -t multi
```


## 3. License

```
MIT License

Copyright (c) 2020 Yantong Lai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```


## 4. Citations

[1] Hovy, D., Johannsen, A., Søgaard, A.: User review sites as a resource for large- scale sociolinguistic studies. In: Proceedings of the 24th international conference on World Wide Web. pp. 452–461 (2015)

[2] Joulin, A., Grave, E., Bojanowski, P., Mikolov, T.: Bag of tricks for efficient text classification. In: Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 2, Short Papers. pp. 427–431. Association for Computational Linguistics (April 2017)
