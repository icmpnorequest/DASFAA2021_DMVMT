# coding=utf8
"""
@author: Yantong Lai
@date: 08/19/2020
@description: Deep multi-view multi-task learning for user gender and age classification.
"""

from torchtext import data
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import spacy

spacy.load("en_core_web_sm")

import time
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# Statistic metric
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel

SEED = 1234

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Dataset path
dataset_path = "../dataset/WWW2015_sep"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--location', type=str, default='denmark', required=True,
                    help='English reviews of different locations')
parser.add_argument('-t', '--task', type=str, default='gender', required=True, help='Task: gender/age/multi')
parser.add_argument('-v', '--view', type=str, default='multi-view', required=False, help='multi')
parser.add_argument('-lamb1', '--mtl_lambda1', type=float, default=1, required=False,
                    help='Lambda for multi-task learning')
parser.add_argument('-lamb2', '--mtl_lambda2', type=float, default=1, required=False,
                    help='Lambda for multi-task learning')
parser.add_argument('-bs', '--batch_size', type=int, default=64, required=False, help='Batch size')
parser.add_argument('-epoch', '--epoch', default=30, required=False, type=int, help='Epoch')
parser.add_argument('-lr', '--learning_rate', default=1e-3, required=False, type=float, help='Learning rate')
parser.add_argument('-shared_hid_dim', '--shared_hidden_dim', default=300, required=False, type=int,
                    help='Shared embedding dim')
parser.add_argument('-max_len', '--max_input_length', default=512, required=False, type=int, help='Max input length')
parser.add_argument('-dropout', '--dropout', default=0.5, required=False, type=float, help='Dropout')
parser.add_argument('-decay', '--weight_decay', default=1e-5, required=False, type=float, help='Weight decay')
parser.add_argument('-ratio', '--train_ratio', type=str, default='100', required=False, help='Ratio to split dataset')
args = parser.parse_args()

# Save result
res_path = "result/" + args.location
if not os.path.exists(res_path):
    os.makedirs(res_path)

res_filename = os.path.join(res_path, "DMVMT.txt")

####################################
#         Hyper-parameters         #
####################################
# Parameters
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
DROPOUT = args.dropout
NUM_EPOCHS = args.epoch
DECAY = args.weight_decay
PATIENCE = 7

# TextCNNs
TOKEN_EMBEDDING_DIM = 200
N_FILTERS = 128
FILTER_SIZES = [2, 3, 4, 5]
FIX_LENGTH = 50

# Shared layer
SHARED_HIDDEN_DIM = args.shared_hidden_dim

# Multi-task learning hyper-parameters
LAMBDA1 = args.mtl_lambda1
LAMBDA2 = args.mtl_lambda2

print("NUM_EPOCHS: ", NUM_EPOCHS)
print("SHARED_HIDDEN_DIM: ", SHARED_HIDDEN_DIM)

####################################
#          Preparing Data          #
####################################
# Bert max input and position token idx
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_input_length = args.max_input_length
print("max_input_length: ", max_input_length)
print("\n")

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens


# 1. data.Field()
TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  tokenize=tokenize_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  init_token=init_token_idx,
                  eos_token=eos_token_idx,
                  pad_token=pad_token_idx,
                  unk_token=unk_token_idx)

SENTI_TOKENS = data.Field(batch_first=True, tokenize=tokenize_and_cut, fix_length=FIX_LENGTH)
TOPIC_TOKENS = data.Field(batch_first=True, tokenize=tokenize_and_cut, fix_length=FIX_LENGTH, )

AGE_LABEL = data.LabelField()
GENDER_LABEL = data.LabelField()
RATING = data.LabelField()
LOCATION_LABEL = data.LabelField()
SENTI_VALUES = data.LabelField()

# 2. data.TabularDataset
if args.train_ratio == '5' or args.train_ratio == '10' or args.train_ratio == '30' or args.train_ratio == '50':
    data_path = os.path.join(dataset_path, args.location + "_total_" + args.train_ratio + ".csv")
else:
    data_path = os.path.join(dataset_path, args.location + "_total.csv")
print("data_path: {}".format(data_path))

dataset = data.TabularDataset(path=data_path, format="csv", fields=[('text', TEXT),
                                                                    ('age', AGE_LABEL),
                                                                    ('gender', GENDER_LABEL),
                                                                    ('rating', RATING),
                                                                    ('location', LOCATION_LABEL),
                                                                    ('sentiment_token', SENTI_TOKENS),
                                                                    ('sentiment_value', SENTI_VALUES),
                                                                    ('topic_token', TOPIC_TOKENS)])
train_data, test_data = dataset.split(split_ratio=0.9)
train_data, valid_data = train_data.split(split_ratio=0.9)

print("Number of train_data = {}".format(len(train_data)))
print("Number of valid_data = {}".format(len(valid_data)))
print("Number of test_data = {}".format(len(test_data)))
print("vars(train_data[0]) = {}\n".format(vars(train_data[0])))

# 3. data.BucketIterator
train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE,
                                                               device=device,
                                                               sort_key=lambda x: len(x.text),
                                                               sort_within_batch=False,
                                                               repeat=False)

# 4. Build vocab
# 4.1 (Optional) If build vocab with pre-trained word embedding vectors
TEXT.build_vocab(train_data)
AGE_LABEL.build_vocab(train_data)
GENDER_LABEL.build_vocab(train_data)
RATING.build_vocab(train_data)
LOCATION_LABEL.build_vocab(train_data)
SENTI_TOKENS.build_vocab(train_data)
SENTI_VALUES.build_vocab(train_data)
TOPIC_TOKENS.build_vocab(train_data)

AGE_OUTPUT_DIM = len(AGE_LABEL.vocab)
GENDER_OUTPUT_DIM = len(GENDER_LABEL.vocab)
LOCATION_OUTPUT_DIM = len(LOCATION_LABEL.vocab)

print("len(TEXT.vocab) = ", len(TEXT.vocab))
print("len(RATING.vocab) = ", len(RATING.vocab))
print("len(SENTI_TOKENS.vocab) = ", len(SENTI_TOKENS.vocab))
print("len(TOPIC_TOKENS).vocab = ", len(TOPIC_TOKENS.vocab))


####################################
#          Build the Model         #
####################################
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss
        # 1) No best_score
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        # 2) score < best_score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        # 3) score >= best_score
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), 'DMVMT_checkpoint.pt')
        self.val_loss_min = val_loss


# Create a BERT instance
BERT = BertModel.from_pretrained('bert-base-uncased')


class MultiviewModel(nn.Module):
    def __init__(self, bert, senti_vocab_size, topic_vocab_size, senti_embed_dim, topic_embed_dim,
                 n_filters, filter_sizes, dropout, shared_hidden_dim, gender_output_dim, age_output_dim):
        super(MultiviewModel, self).__init__()

        # 1. Text Model
        # 1) BERT for text embedding
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        # embedding_dim = 768

        # 2. Sentiment Model
        # 1) Look-up embedding
        self.senti_embedding = nn.Embedding(num_embeddings=senti_vocab_size, embedding_dim=senti_embed_dim)
        # 2) Sentiment token TextCNN layer
        self.senti_convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, senti_embed_dim))
            for fs in filter_sizes])

        # 3. Topic Model
        # 1) Embedding
        self.topic_embedding = nn.Embedding(num_embeddings=topic_vocab_size, embedding_dim=topic_embed_dim)
        # 2) Topic token TextCNN layer
        self.topic_convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, topic_embed_dim))
            for fs in filter_sizes])

        # 4. Shared representation layer
        self.fc = nn.Linear(in_features=embedding_dim + 2 * len(filter_sizes) * n_filters,
                            out_features=shared_hidden_dim)

        # 5. Gender task
        self.fc_gender = nn.Linear(shared_hidden_dim, gender_output_dim)

        # 6. Age task
        self.fc_age = nn.Linear(shared_hidden_dim, age_output_dim)

        # 7. Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, text, senti_words, topic_words):
        # 1. Encode review
        # 1) Embed text.
        # text = [batch size, sent len]
        with torch.no_grad():
            # Get the hidden state of BERT's last layer
            embedded = self.bert(text)[0]
        avg_embedded = torch.mean(embedded, dim=1)
        # avg_embedded = [batch size, 768]

        # 2. Sentiment words TextCNN
        # 1) Embed sentiment tokens
        embedded_senti_token = self.senti_embedding(senti_words)
        # embedded_senti_token = [batch size, sent len, emb dim]

        embedded_senti_token = embedded_senti_token.unsqueeze(1)
        # embedded_senti_token = [batch size, 1, sent len, emb dim]

        # 2) Convolution
        senti_conved = [F.relu(conv(embedded_senti_token)).squeeze(3) for conv in self.senti_convs]
        # senti_conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # 3) Max-pooling
        senti_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in senti_conved]
        # senti_pooled_n = [batch size, n_filters]

        # 4) Concat all max-pooling layer
        senti_cat = self.dropout(torch.cat(senti_pooled, dim=1))
        # senti_cat = [batch size, n_filters * len(filter_sizes)]

        # 3. Topic words TextCNN
        # 1) Embed topic tokens
        embedded_topic_token = self.topic_embedding(topic_words)
        # embedded_topic_token = [batch size, sent len, emb dim]

        embedded_topic_token = embedded_topic_token.unsqueeze(1)
        # embedded_topic_token = [batch size, 1, sent len, emb dim]

        # 2) Convolution
        topic_conved = [F.relu(conv(embedded_topic_token)).squeeze(3) for conv in self.topic_convs]
        # topic_conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        # 3) Max-pooling
        topic_pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in topic_conved]
        # topic_pooled_n = [batch size, n_filters]

        # 4) Concat all max-pooling layer
        topic_cat = self.dropout(torch.cat(topic_pooled, dim=1))
        # topic_cat = [batch size, n_filters * len(filter_sizes)]

        # 5. Concat
        # total_cat = torch.cat([embedded, senti_cat, topic_cat], dim=1)
        total_cat = torch.cat((avg_embedded, senti_cat, topic_cat), dim=1)
        # total_cat = [batch size, 768 + 2 * n_filters * len(filter_sizes)]

        # 6. FC layer: reduce dimensions
        h_shared = self.fc(total_cat)
        # h_shared = [batch size, shared_hidden_dim]

        # 7. Gender output
        gender_output = self.fc_gender(h_shared)
        gender_output = F.softmax(gender_output)

        # 8. Age output
        age_output = self.fc_age(h_shared)
        age_output = F.softmax(age_output)

        return gender_output, age_output


# Parameters
INPUT_DIM = len(TEXT.vocab)
print("INPUT_DIM = {}\n".format(INPUT_DIM))

# Create an instance
model = MultiviewModel(bert=BERT, senti_vocab_size=len(SENTI_TOKENS.vocab), topic_vocab_size=len(TOPIC_TOKENS.vocab),
                       senti_embed_dim=TOKEN_EMBEDDING_DIM, topic_embed_dim=TOKEN_EMBEDDING_DIM,
                       n_filters=N_FILTERS, filter_sizes=FILTER_SIZES, dropout=DROPOUT,
                       shared_hidden_dim=SHARED_HIDDEN_DIM, gender_output_dim=GENDER_OUTPUT_DIM,
                       age_output_dim=AGE_OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=DECAY)
model = model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss().to(device)

####################################
#          Train the Model         #
####################################
# Train
total_step = len(train_iter)
print("total_step = {}\n".format(total_step))


def train_valid(model, task):
    """
    It is a function to train and validation the model
    """
    # Define <List> object to save loss, avg_loss & acc
    # 1. Loss & avg loss
    total_loss = []
    total_avg_loss = []

    total_gender_loss = []
    total_avg_gender_loss = []

    total_age_loss = []
    total_avg_age_loss = []

    # 2. Statistic metrics
    # 1) Accuracy
    total_acc = []
    gender_acc = []
    age_acc = []

    # 2) F1-macro
    gender_macrof1 = []
    age_macrof1 = []
    total_macrof1 = []

    # 3. Probability metrics
    y_gender_proba_list = []
    y_age_proba_list = []

    # 4. Validation list
    valid_gender_acc_list = []
    valid_age_acc_list = []
    valid_multi_acc_list = []

    valid_gender_avg_loss_list = []
    valid_age_avg_loss_list = []
    valid_multi_avg_loss_list = []

    valid_gender_macro_f1_list = []
    valid_age_macro_f1_list = []
    valid_multi_macro_f1_list = []

    # Initialize the early_stopping object
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    for epoch in range(NUM_EPOCHS):
        print("##### Epoch {} / {} #####".format(epoch + 1, NUM_EPOCHS))

        # Measure how long the training epoch takes.
        t0 = time.time()
        train_total_loss = 0
        train_gender_loss = 0
        train_age_loss = 0
        train_total_correct = 0
        train_gender_correct = 0
        train_age_correct = 0

        y_gender_list = []
        y_age_list = []
        y_total_list = []
        pred_gender_list = []
        pred_age_list = []
        pred_total_list = []

        # Start training
        model.train()
        for i, batch in enumerate(train_iter):
            # Clear
            optimizer.zero_grad()

            # Get text
            text = batch.text.to(device)
            sentiment_token = batch.sentiment_token.to(device)
            topic_token = batch.topic_token.to(device)

            if task == 'gender':
                # Get label
                y_gender = batch.gender.to(device)
                y_gender_list.append(y_gender)

                # Forward pass, return probability
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)
                y_gender_proba_list.append(y_gender_pred)

                # Loss function
                loss_gender = criterion(y_gender_pred, y_gender)

                # Gender prediction
                pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                pred_gender_list.append(pred_gender)

                # Get the correct number
                train_gender_correct += (pred_gender == y_gender).sum().item()

                # Backward and optimize
                loss_gender.backward()
                optimizer.step()

                # Get gender loss
                total_gender_loss.append(loss_gender.item())
                train_gender_loss += loss_gender.item()
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                          .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss_gender.item(),
                                  accuracy_score(y_gender.data.cpu().numpy(), pred_gender.data.cpu().numpy())))

            elif task == 'age':
                # Get label
                y_age = batch.age.to(device)
                y_age_list.append(y_age)

                # Forward pass, return probability
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)
                y_age_proba_list.append(y_age_pred)

                # Loss function
                loss_age = criterion(y_age_pred, y_age)

                # Age prediction
                pred_age = torch.argmax(y_age_pred.data, dim=1)
                pred_age_list.append(pred_age)

                # Get the correct number
                train_age_correct += (pred_age == y_age).sum().item()

                # Backward and optimize
                loss_age.backward()
                optimizer.step()

                # Get age loss
                total_age_loss.append(loss_age.item())
                train_age_loss += loss_age.item()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.4f}'
                          .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss_age.item(),
                                  accuracy_score(y_age.data.cpu().numpy(), pred_age.data.cpu().numpy())))

            elif task == 'multi':
                # Get label
                y_gender = batch.gender.to(device)
                y_age = batch.age.to(device)
                y_total = torch.cat([y_gender, y_age])

                # Add to <List> objects
                y_gender_list.append(y_gender)
                y_age_list.append(y_age)
                y_total_list.append(y_total)

                # Forward pass, return probability
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                y_gender_proba_list.append(y_gender_pred)
                y_age_proba_list.append(y_age_pred)

                # Loss function
                loss_gender = criterion(y_gender_pred, y_gender)
                loss_age = criterion(y_age_pred, y_age)

                loss = LAMBDA1 * loss_gender + LAMBDA2 * loss_age

                # Multi-task prediction
                pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                pred_age = torch.argmax(y_age_pred.data, dim=1)
                pred_total = torch.cat([pred_gender, pred_age])
                pred_gender_list.append(pred_gender)
                pred_age_list.append(pred_age)
                pred_total_list.append(pred_total)

                # Get the correct number
                train_gender_correct += (pred_gender == y_gender).sum().item()
                train_age_correct += (pred_age == y_age).sum().item()
                train_total_correct += (pred_total == y_total).sum().item()

                # Backward and optimize
                loss.backward()
                optimizer.step()

                # Add multi-task loss
                total_gender_loss.append(loss_gender.item())
                total_age_loss.append(loss_age.item())
                total_loss.append(loss.item())

                train_gender_loss += loss_gender.item()
                train_age_loss += loss_age.item()
                train_total_loss += loss.item()

                if (i + 1) % 100 == 0:
                    print(
                        'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Gender Acc: {:4f}, Age Acc: {:4f}, Total Acc: {:4f}'
                            .format(epoch + 1, NUM_EPOCHS, i + 1, total_step, loss.item(),
                                    accuracy_score(y_gender.data.cpu().numpy(), pred_gender.data.cpu().numpy()),
                                    accuracy_score(y_age.data.cpu().numpy(), pred_age.data.cpu().numpy()),
                                    accuracy_score(y_total.data.cpu().numpy(), pred_total.data.cpu().numpy())))

        # Print result after training a batch
        if task == 'gender':
            # 1. Concat tensors in a list
            # y_label
            y_gender_ = torch.cat(y_gender_list)
            # y_pred
            pred_gender_ = torch.cat(pred_gender_list)

            # 2. Statistics result
            # 1) Avg gender loss
            gender_avg_loss = train_gender_loss / len(train_data)
            total_avg_gender_loss.append(gender_avg_loss)
            print("Avg train gender loss = {}".format(gender_avg_loss))

            # 2) Accuracy
            gender_acc.append(train_gender_correct / len(train_data))
            print("Training gender accuracy = {}".format(train_gender_correct / len(train_data)))

            # 3) F1-score, (y_label, y_pred)
            gender_macrof1.append(
                f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(), average='macro'))
            print("Training gender F1-macro: {}".format(
                f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(), average='macro')))

            # 4) Time spending
            print("Training epcoh took: {}s\n".format(time.time() - t0))

        elif task == 'age':
            # 1. Concat tensors in a list
            # y_label
            y_age_ = torch.cat(y_age_list)
            # y_pred
            pred_age_ = torch.cat(pred_age_list)

            # 2. Statistics result
            # 1) Avg age loss
            age_avg_loss = train_age_loss / len(train_data)
            total_avg_age_loss.append(age_avg_loss)
            print("Avg train age loss = {}".format(age_avg_loss))

            # 2) Accuracy
            age_acc.append(train_age_correct / len(train_data))
            print("Training age accuracy = {}".format(train_age_correct / len(train_data)))

            # 3) F1-score, (y_label, y_pred)
            age_macrof1.append(
                f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro'))
            print("Training age F1-macro: {}".format(
                f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro')))

            # 4) Time spending
            print("Training epcoh took: {}s\n".format(time.time() - t0))

        elif task == 'multi':
            # 1. Concat tensors in a list
            # 1) y_label
            y_gender_ = torch.cat(y_gender_list)
            y_age_ = torch.cat(y_age_list)
            y_total_ = torch.cat(y_total_list)

            # 2) y_pred
            pred_gender_ = torch.cat(pred_gender_list)
            pred_age_ = torch.cat(pred_age_list)
            pred_total_ = torch.cat(pred_total_list)

            # 2. Statistics result
            # 1) Avg total loss
            gender_avg_loss = train_gender_loss / len(train_data)
            age_avg_loss = train_age_loss / len(train_data)
            avg_loss = train_total_loss / len(train_data)

            total_avg_gender_loss.append(gender_avg_loss)
            total_avg_age_loss.append(age_avg_loss)
            total_avg_loss.append(avg_loss)
            print("Avg train gender loss = {}".format(gender_avg_loss))
            print("Avg train age loss = {}".format(age_avg_loss))
            print("Avg train total loss = {}".format(avg_loss))

            # 2) Accuracy
            gender_acc.append(train_gender_correct / len(train_data))
            age_acc.append(train_age_correct / len(train_data))
            total_acc.append(train_total_correct / (2 * len(train_data)))
            print("Training gender accuracy = {}".format(train_gender_correct / len(train_data)))
            print("Training age accuracy = {}".format(train_age_correct / len(train_data)))
            print("Training total accuracy = {}".format(train_total_correct / (2 * len(train_data))))

            # 3) F1-score, (y_label, y_pred)
            gender_macrof1.append(
                f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(), average='macro'))
            print("Training gender F1-macro: {}".format(
                f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(), average='macro')))
            age_macrof1.append(
                f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro'))
            print("Training age F1-macro: {}".format(
                f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro')))
            total_macrof1.append(
                f1_score(y_total_.detach().cpu().numpy(), pred_total_.detach().cpu().numpy(), average='macro'))
            print("Training total F1-macro: {}".format(
                f1_score(y_total_.detach().cpu().numpy(), pred_total_.detach().cpu().numpy(), average='macro')))

            # 4) Time spending
            print("Training epcoh took: {}s\n".format(time.time() - t0))

        # 2. Validation
        valid_total_correct = 0
        valid_total_loss = 0.0
        valid_gender_loss = 0.0
        valid_age_loss = 0.0
        valid_gender_correct = 0
        valid_age_correct = 0

        # Statistic
        valid_y_gender_list = []
        valid_y_age_list = []
        valid_y_total_list = []
        valid_pred_gender_list = []
        valid_pred_age_list = []
        valid_pred_total_list = []

        with torch.no_grad():
            model.eval()

            for i, batch in enumerate(valid_iter):
                # Get text
                text = batch.text.to(device)
                sentiment_token = batch.sentiment_token.to(device)
                topic_token = batch.topic_token.to(device)

                if task == 'gender':
                    # Get label
                    y_gender = batch.gender.to(device)
                    valid_y_gender_list.append(y_gender)

                    # Predict gender
                    y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                    # Loss
                    loss_gender = criterion(y_gender_pred, y_gender)

                    # Get gender prediction
                    pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                    valid_pred_gender_list.append(pred_gender)

                    # Get correct number & loss
                    valid_gender_correct += (pred_gender == y_gender).sum().item()
                    valid_total_loss += loss_gender.item()

                elif task == 'age':
                    # Get label
                    y_age = batch.age.to(device)
                    valid_y_age_list.append(y_age)

                    # Predict age
                    y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                    # Loss
                    loss_age = criterion(y_age_pred, y_age)

                    # Get age prediction
                    pred_age = torch.argmax(y_age_pred.data, dim=1)
                    valid_pred_age_list.append(pred_age)

                    # Get correct number & loss
                    valid_age_correct += (pred_age == y_age).sum().item()
                    valid_total_loss += loss_age.item()

                elif task == 'multi':
                    # Get label
                    y_gender = batch.gender.to(device)
                    y_age = batch.age.to(device)
                    y_total = torch.cat([y_gender, y_age])
                    valid_y_gender_list.append(y_gender)
                    valid_y_age_list.append(y_age)
                    valid_y_total_list.append(y_total)

                    # Predict gender & age
                    y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                    # Loss
                    loss_gender = criterion(y_gender_pred, y_gender)
                    loss_age = criterion(y_age_pred, y_age)
                    loss = LAMBDA1 * loss_gender + LAMBDA2 * loss_age

                    # Get gender & age prediction
                    pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                    pred_age = torch.argmax(y_age_pred.data, dim=1)
                    pred_total = torch.cat([pred_gender, pred_age])
                    valid_pred_gender_list.append(pred_gender)
                    valid_pred_age_list.append(pred_age)
                    valid_pred_total_list.append(pred_total)

                    # Get correct number
                    valid_gender_correct += (pred_gender == y_gender).sum().item()
                    valid_age_correct += (pred_age == y_age).sum().item()
                    valid_total_correct += (pred_total == y_total).sum().item()
                    valid_total_loss += loss.item()
                    valid_gender_loss += loss_gender.item()
                    valid_age_loss += loss_age.item()

        # Print validation result
        print("##### Validation #####")
        if task == 'gender':
            # 1. Concat tensors in a list
            # y_label
            y_gender_ = torch.cat(valid_y_gender_list)
            # y_pred
            pred_gender_ = torch.cat(valid_pred_gender_list)

            # 2. Statistics result
            # 1) Avg gender loss
            avg_loss = valid_total_loss / (2 * len(valid_data))
            valid_gender_avg_loss_list.append(avg_loss)
            print("Valid gender avg loss: {}".format(avg_loss))

            # 2) Accuracy
            valid_gender_acc = valid_gender_correct / len(valid_data)
            valid_gender_acc_list.append(valid_gender_acc)
            print("Valid gender accuracy: {}".format(valid_gender_acc))

            # 3) F1-score, (y_label, y_pred)
            valid_gender_macro_f1 = f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(),
                                             average='macro')
            valid_gender_macro_f1_list.append(valid_gender_macro_f1)
            print("Valid gender F1-macro: {}\n".format(valid_gender_macro_f1))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if epoch >= 10:
                early_stopping(val_loss=avg_loss, model=model)
                if early_stopping.early_stop:
                    print("Gender model early stopping")
                    break

        elif task == 'age':
            # 1. Concat tensors in a list
            # y_label
            y_age_ = torch.cat(valid_y_age_list)
            # y_pred
            pred_age_ = torch.cat(valid_pred_age_list)

            # 2. Statistics result
            # 1) Avg age loss
            avg_loss = valid_total_loss / (2 * len(valid_data))
            valid_age_avg_loss_list.append(avg_loss)
            print("Valid age avg loss: {}".format(avg_loss))

            # 2) Accuracy
            valid_age_acc = valid_age_correct / len(valid_data)
            valid_age_acc_list.append(valid_age_acc)
            print("Valid age accuracy: {}".format(valid_age_acc))

            # 3) F1-score, (y_label, y_pred)
            valid_age_macro_f1 = f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(),
                                          average='macro')
            valid_age_macro_f1_list.append(valid_age_macro_f1)
            print("Valid age F1-macro: {}\n".format(valid_age_macro_f1))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if epoch >= 10:
                early_stopping(val_loss=avg_loss, model=model)
                if early_stopping.early_stop:
                    print("Age model early stopping")
                    break

        elif task == 'multi':
            # 1. Concat tensors in a list
            # y_label
            y_gender_ = torch.cat(valid_y_gender_list)
            y_age_ = torch.cat(valid_y_age_list)
            y_total_ = torch.cat(valid_y_total_list)

            # y_pred
            pred_gender_ = torch.cat(valid_pred_gender_list)
            pred_age_ = torch.cat(valid_pred_age_list)
            pred_total_ = torch.cat(valid_pred_total_list)

            # 2. Statistics result
            # 1) Avg total loss
            avg_loss = valid_total_loss / (2 * len(valid_data))
            valid_multi_avg_loss_list.append(avg_loss)
            avg_gender_loss = valid_gender_loss / len(valid_data)
            valid_gender_avg_loss_list.append(avg_gender_loss)
            avg_age_loss = valid_age_loss / len(valid_data)
            valid_age_avg_loss_list.append(avg_age_loss)
            print("Valid gender avg loss: {}".format(avg_gender_loss))
            print("Valid age avg loss: {}".format(avg_age_loss))
            print("Valid multi avg loss: {}".format(avg_loss))

            # 2) Accuracy
            valid_gender_acc = valid_gender_correct / len(valid_data)
            valid_age_acc = valid_age_correct / len(valid_data)
            valid_total_acc = valid_total_correct / (2 * len(valid_data))
            valid_gender_acc_list.append(valid_gender_acc)
            valid_age_acc_list.append(valid_age_acc)
            valid_multi_acc_list.append(valid_total_acc)
            print("Validation gender accuracy = {}".format(valid_gender_acc))
            print("Validation age accuracy = {}".format(valid_age_acc))
            print("Validation total accuracy: {}".format(valid_total_acc))

            # 3) F1-score, (y_label, y_pred)
            valid_gender_macro_f1 = f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(),
                                             average='macro')
            print("Valid gender F1-macro: {}".format(valid_gender_macro_f1))
            valid_age_macro_f1 = f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(),
                                          average='macro')
            print("Valid age F1-macro: {}".format(valid_age_macro_f1))
            valid_total_macro_f1 = f1_score(y_total_.detach().cpu().numpy(), pred_total_.detach().cpu().numpy(),
                                            average='macro')
            print("Valid total F1-macro: {}\n".format(valid_total_macro_f1))

            valid_gender_macro_f1_list.append(valid_gender_macro_f1)
            valid_age_macro_f1_list.append(valid_age_macro_f1)
            valid_multi_macro_f1_list.append(valid_total_macro_f1)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if epoch >= 10:
                early_stopping(val_loss=avg_loss, model=model)
                if early_stopping.early_stop:
                    print("Multi-task model early stops")
                    break

    # Return training results
    model.load_state_dict(torch.load('DMVMT_checkpoint.pt'))
    if task == 'gender':
        # load the last checkpoint with the best model
        # model.load_state_dict(torch.load('DMVMT_checkpoint.pt'))
        return total_avg_gender_loss, gender_acc, gender_macrof1, model, valid_gender_avg_loss_list, \
               valid_gender_acc_list, valid_gender_macro_f1_list
    elif task == 'age':
        # model.load_state_dict(torch.load('DMVMT_checkpoint.pt'))
        return total_avg_age_loss, age_acc, age_macrof1, model, valid_age_avg_loss_list, valid_age_acc_list, \
               valid_age_macro_f1_list
    elif task == 'multi':
        # model.load_state_dict(torch.load('DMVMT_checkpoint.pt'))
        return total_avg_gender_loss, total_avg_age_loss, total_avg_loss, gender_acc, age_acc, total_acc, gender_macrof1, \
               age_macrof1, total_macrof1, model, valid_gender_avg_loss_list, valid_age_avg_loss_list, \
               valid_multi_avg_loss_list, valid_gender_acc_list, valid_age_acc_list, valid_multi_acc_list, \
               valid_gender_macro_f1_list, valid_age_macro_f1_list, valid_multi_macro_f1_list


####################################
#          Plot the Loss           #
####################################
# Define image path
image_path = "image/" + args.location + "/DMVMT"
if not os.path.exists(image_path):
    os.makedirs(image_path)


def plot_avg_loss(image_path, task, location, view, total_avg_gender_loss, total_avg_age_loss, total_avg_loss,
                  mode="train"):
    """
    It is a function to plot the training avg loss.
    """
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (15, 10)

    fig = plt.gcf()
    if task == 'gender':
        # Plot the learning curve.
        plt.plot(range(1, len(total_avg_gender_loss) + 1), total_avg_gender_loss, label=mode + "_avg_gender_loss")

    elif task == 'age':
        plt.plot(range(1, len(total_avg_age_loss) + 1), total_avg_age_loss, label=mode + "_avg_age_loss")

    elif task == 'multi':
        plt.plot(range(1, len(total_avg_gender_loss) + 1), total_avg_gender_loss, label=mode + "_avg_gender_loss")
        plt.plot(range(1, len(total_avg_age_loss) + 1), total_avg_age_loss, label=mode + "_avg_age_loss")
        plt.plot(range(1, len(total_avg_loss) + 1), total_avg_loss, label=mode + "_avg_loss")

    # Label the plot.
    image_name = "Location: " + location + ", Task: " + task + ", View: " + view + ", Epoch: " + str(
        NUM_EPOCHS) + ", Shared Hidden Dim: " + str(SHARED_HIDDEN_DIM) + ", LAMBDA1: " + str(LAMBDA1) + ", LAMBDA1: " \
                 + str(LAMBDA1) + ", DMVMT Avg Loss " + mode
    plt.title(image_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.xticks(np.arange(0, NUM_EPOCHS + 1, 5.0))
    plt.legend()
    # plt.show()
    # plt.draw()
    fig.savefig(os.path.join(image_path, image_name + ".png"), dpi=100)
    plt.close()


def plot_acc(image_path, task, location, view, gender_acc, age_acc, total_acc, mode="train"):
    """
    It is a function to plot accuracy.
    """
    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    fig = plt.gcf()

    if task == 'gender':
        # Plot the learning curve.
        # plt.plot(range(1, NUM_EPOCHS + 1), gender_acc, '-o', label="train_gender_acc")
        plt.plot(range(1, len(gender_acc) + 1), gender_acc, '-o', label=mode + "_gender_acc")

    elif task == 'age':
        # plt.plot(range(1, NUM_EPOCHS + 1), age_acc, '-o', label="train_age_acc")
        plt.plot(range(1, len(age_acc) + 1), age_acc, '-o', label=mode + "_age_acc")

    elif task == 'multi':
        # plt.plot(range(1, NUM_EPOCHS + 1), gender_acc, '-o', label="train_gender_acc")
        plt.plot(range(1, len(gender_acc) + 1), gender_acc, '-o', label=mode + "_gender_acc")
        # plt.plot(range(1, NUM_EPOCHS + 1), age_acc, '-o', label="train_age_acc")
        plt.plot(range(1, len(age_acc) + 1), age_acc, '-o', label=mode + "_age_acc")
        # plt.plot(range(1, NUM_EPOCHS + 1), total_acc, '-o', label="train_total_acc")
        plt.plot(range(1, len(total_acc) + 1), total_acc, '-o', label=mode + "_total_acc")

    # Label the plot.
    image_name = "Location: " + location + ", Task: " + task + ", View: " + view + ", Epoch: " + str(
        NUM_EPOCHS) + ", Shared Hidden Dim: " + str(SHARED_HIDDEN_DIM) + ", LAMBDA1: " + str(LAMBDA1) + ", LAMBDA1: " \
                 + str(LAMBDA1) + ", DMVMT Acc " + mode
    plt.title(image_name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.xticks(np.arange(0, NUM_EPOCHS + 1, 5.0))
    plt.legend()
    # plt.show()
    plt.draw()
    fig.savefig(os.path.join(image_path, image_name + ".png"), dpi=100)
    plt.close()


####################################
#         Save the Results         #
####################################
def save_training_result(location, task, view, filename, total_avg_gender_loss, total_avg_age_loss, total_avg_loss,
                         gender_acc, age_acc, total_acc, gender_macrof1, age_macrof1, total_macrof1):
    """
    It is a function to save training result into txt file.
    """
    start_line = "##### Location: " + location + ", Task: " + task + ", View: " + view + ", Epoch: " + str(
        NUM_EPOCHS) + ", Dropout: " + str(DROPOUT) + ", SHARED_HIDDEN_DIM: " + str(SHARED_HIDDEN_DIM) + ", LAMBDA: " \
                 + str(LAMBDA1) + ", LAMBDA2: " + str(LAMBDA2) + ", Batch size: " + str(BATCH_SIZE) + "Learning rate: " \
                 + str(LEARNING_RATE) + " #####\n"
    with open(filename, 'a') as f:
        f.write(start_line)

        if task == 'gender':
            f.write("Training total_avg_gender_loss: ")
            f.writelines(', '.join(str(item) for item in total_avg_gender_loss))
            f.write("\n")
            f.write("Training gender_acc: ")
            f.writelines(', '.join(str(item) for item in gender_acc))
            f.write("\n")
            f.write("Training gender_f1-macro: ")
            f.writelines(', '.join(str(item) for item in gender_macrof1))
            f.write("\n")

        elif task == 'age':
            f.write("Training total_avg_age_loss: ")
            f.writelines(', '.join(str(item) for item in total_avg_age_loss))
            f.write("\n")
            f.write("Training age_acc: ")
            f.writelines(', '.join(str(item) for item in age_acc))
            f.write("\n")
            f.write("Training age_f1-macro: ")
            f.writelines(', '.join(str(item) for item in age_macrof1))
            f.write("\n")

        elif task == 'multi':
            # 1) Write gender result
            f.write("Training total_avg_gender_loss: ")
            f.writelines(', '.join(str(item) for item in total_avg_gender_loss))
            f.write("\n")
            f.write("Training gender_acc: ")
            f.writelines(', '.join(str(item) for item in gender_acc))
            f.write("\n")
            f.write("Training gender_f1-macro: ")
            f.writelines(', '.join(str(item) for item in gender_macrof1))
            f.write("\n")

            # 2) Write age result
            f.write("Training total_avg_age_loss: ")
            f.writelines(', '.join(str(item) for item in total_avg_age_loss))
            f.write("\n")
            f.write("Training age_acc: ")
            f.writelines(', '.join(str(item) for item in age_acc))
            f.write("\n")
            f.write("Training age_f1-macro: ")
            f.writelines(', '.join(str(item) for item in age_macrof1))
            f.write("\n")

            # 3) Write total result
            f.write("Training total_avg_loss: ")
            f.writelines(', '.join(str(item) for item in total_avg_loss))
            f.write("\n")
            f.write("Training total_acc: ")
            f.writelines(', '.join(str(item) for item in total_acc))
            f.write("\n")
            f.write("Training total_f1-macro: ")
            f.writelines(', '.join(str(item) for item in total_macrof1))
            f.write("\n")


def save_test_result(filename, task, avg_loss, test_gender_acc, test_age_acc, test_total_acc, test_gender_macrof1,
                     test_age_macrof1, test_total_macrof1):
    """
    It is a function to save test results.
    """
    with open(filename, 'a') as f:
        if task == 'gender':
            # 1) loss
            f.write("Test avg_loss: ")
            f.write(str(avg_loss) + "\n")
            # 2) gender acc
            f.write("Test test_gender_acc: ")
            f.write(str(test_gender_acc) + "\n")
            # 3) F1-macro
            f.write("Test test_gender_macrof1: ")
            f.write(str(test_gender_macrof1) + "\n")
            f.write("\n")

        elif task == 'age':
            # 1) loss
            f.write("Test avg_loss: ")
            f.write(str(avg_loss) + "\n")
            # 2) age acc
            f.write("Test test_age_acc: ")
            f.write(str(test_age_acc) + "\n")
            # 3) F1-macro
            f.write("Test test_age_macrof1: ")
            f.write(str(test_age_macrof1) + "\n")
            f.write("\n")

        elif task == 'multi':
            # 1) loss
            f.write("Test avg_loss: ")
            f.write(str(avg_loss) + "\n")
            # 2) gender acc
            f.write("Test test_gender_acc: ")
            f.write(str(test_gender_acc) + "\n")
            f.write("Test test_age_acc: ")
            f.write(str(test_age_acc) + "\n")
            f.write("Test test_total_acc: ")
            f.write(str(test_total_acc) + "\n")
            # 3) F1-macro
            f.write("Test test_gender_macrof1: ")
            f.write(str(test_gender_macrof1) + "\n")
            f.write("Test test_age_macrof1: ")
            f.write(str(test_age_macrof1) + "\n")
            f.write("Test test_total_macrof1: ")
            f.write(str(test_total_macrof1) + "\n")
            f.write("\n")
            f.write("\n")


# Train and Validation
if args.task == 'gender':
    total_avg_gender_loss, gender_acc, gender_macrof1, model, valid_gender_avg_loss_list, \
    valid_gender_acc_list, valid_gender_macro_f1_list = train_valid(model=model, task=args.task)

    print("##### After training results #####")
    print("Training total_avg_gender_loss: {}".format(total_avg_gender_loss))
    print("Training gender_acc: {}".format(gender_acc))
    print("Validation gender avg loss: {}".format(valid_gender_avg_loss_list))
    print("Validation gender acc: {}\n".format(valid_gender_acc_list))

    # 1. Plot average loss
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=total_avg_gender_loss, total_avg_age_loss=None, total_avg_loss=None,
                  mode="train")
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=valid_gender_avg_loss_list, total_avg_age_loss=None, total_avg_loss=None,
                  mode="valid")

    # 2. Plot accuracy
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=gender_acc, age_acc=None, total_acc=None, mode="train")
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=valid_gender_acc_list, age_acc=None, total_acc=None, mode="valid")

    # 3. Save the training avg loss & acc
    save_training_result(location=args.location, task=args.task, view=args.view, filename=res_filename,
                         total_avg_gender_loss=total_avg_gender_loss, total_avg_age_loss=None, total_avg_loss=None,
                         gender_acc=gender_acc, age_acc=None, total_acc=None, gender_macrof1=gender_macrof1,
                         age_macrof1=None, total_macrof1=None)
    print("Save training results successfully.\n")

elif args.task == 'age':
    total_avg_age_loss, age_acc, age_macrof1, model, valid_age_avg_loss_list, valid_age_acc_list, \
    valid_age_macro_f1_list = train_valid(model=model, task=args.task)

    print("##### After training results #####")
    print("Training total_avg_age_loss: {}".format(total_avg_age_loss))
    print("Training age_acc: {}".format(age_acc))
    print("Validation age avg loss: {}".format(valid_age_avg_loss_list))
    print("Validation age acc: {}\n".format(valid_age_acc_list))

    # 1. Plot average loss
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=None, total_avg_age_loss=total_avg_age_loss, total_avg_loss=None, mode="train")
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=None, total_avg_age_loss=valid_age_avg_loss_list, total_avg_loss=None,
                  mode="valid")

    # 2. Plot accuracy
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=None, age_acc=age_acc, total_acc=None, mode="train")
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=None, age_acc=valid_age_acc_list, total_acc=None, mode="valid")

    # 3. Save the training avg loss & acc
    save_training_result(location=args.location, task=args.task, view=args.view, filename=res_filename,
                         total_avg_gender_loss=None, total_avg_age_loss=total_avg_age_loss, total_avg_loss=None,
                         gender_acc=None, age_acc=age_acc, total_acc=None, gender_macrof1=None,
                         age_macrof1=age_macrof1, total_macrof1=None)
    print("Save training results successfully.\n")

elif args.task == 'multi':
    total_avg_gender_loss, total_avg_age_loss, total_avg_loss, gender_acc, age_acc, total_acc, gender_macrof1, \
    age_macrof1, total_macrof1, model, valid_gender_avg_loss_list, valid_age_avg_loss_list, \
    valid_multi_avg_loss_list, valid_gender_acc_list, valid_age_acc_list, valid_multi_acc_list, \
    valid_gender_macro_f1_list, valid_age_macro_f1_list, valid_multi_macro_f1_list \
        = train_valid(model=model, task=args.task)

    print("##### After training results #####")
    print("Training total_avg_gender_loss: {}".format(total_avg_gender_loss))
    print("Training total_avg_age_loss: {}".format(total_avg_age_loss))
    print("Training total_avg_loss: {}\n".format(total_avg_loss))

    print("Training gender_acc: {}".format(gender_acc))
    print("Training age_acc: {}".format(age_acc))
    print("Training total_acc: {}\n".format(total_acc))

    print("Valid valid_avg_gender_loss: {}".format(valid_gender_avg_loss_list))
    print("Valid valid_avg_age_loss: {}".format(valid_age_avg_loss_list))
    print("Valid valid_multi_avg_loss: {}\n".format(valid_multi_avg_loss_list))

    print("Valid valid_gender_acc: {}".format(valid_gender_acc_list))
    print("Valid age_acc: {}".format(valid_age_acc_list))
    print("Valid total_acc: {}\n".format(valid_multi_acc_list))

    # 1. Plot average loss
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=total_avg_gender_loss, total_avg_age_loss=total_avg_age_loss,
                  total_avg_loss=total_avg_loss, mode="train")
    plot_avg_loss(image_path=image_path, task=args.task, location=args.location, view=args.view,
                  total_avg_gender_loss=valid_gender_avg_loss_list, total_avg_age_loss=valid_age_avg_loss_list,
                  total_avg_loss=valid_multi_avg_loss_list, mode="valid")

    # 2. Plot accuracy
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=gender_acc, age_acc=age_acc, total_acc=total_acc, mode="train")
    plot_acc(image_path=image_path, task=args.task, location=args.location, view=args.view,
             gender_acc=valid_gender_acc_list, age_acc=valid_age_acc_list, total_acc=valid_multi_acc_list, mode="valid")

    # 3. Save the training avg loss & acc
    save_training_result(location=args.location, task=args.task, view=args.view, filename=res_filename,
                         total_avg_gender_loss=total_avg_gender_loss, total_avg_age_loss=total_avg_age_loss,
                         total_avg_loss=total_avg_loss, gender_acc=gender_acc, age_acc=age_acc, total_acc=total_acc,
                         gender_macrof1=gender_macrof1, age_macrof1=age_macrof1, total_macrof1=total_macrof1)
    print("Save training results successfully.\n")


####################################
#            Evaluation            #
####################################
def evaluate(model, task):
    """
    It is a function to evaluate test dataset.
    """
    model.eval()

    # Definition
    test_gender_correct = 0
    test_age_correct = 0
    test_total_correct = 0
    test_total_loss = 0.0

    # Statistic metrics
    # Label metrics
    y_gender_list = []
    y_age_list = []
    y_total_list = []

    # Prediction metrics
    pred_gender_list = []
    pred_age_list = []
    pred_total_list = []

    with torch.no_grad():
        for i, batch in enumerate(test_iter):

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Get text
            text = batch.text.to(device)
            sentiment_token = batch.sentiment_token.to(device)
            topic_token = batch.topic_token.to(device)

            if task == 'gender':
                # Get label
                y_gender = batch.gender.to(device)
                y_gender_list.append(y_gender)

                # Forward pass, return probability
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                # Loss
                loss_gender = criterion(y_gender_pred, y_gender)

                # Get prediction
                pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                pred_gender_list.append(pred_gender)

                # Get correct number
                test_gender_correct += (pred_gender == y_gender).sum().item()
                test_total_loss += loss_gender.item()

            elif task == 'age':
                # Get label
                y_age = batch.age.to(device)
                y_age_list.append(y_age)

                # Forward pass
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                # Loss
                loss_age = criterion(y_age_pred, y_age)

                # Get prediction
                pred_age = torch.argmax(y_age_pred.data, dim=1)
                pred_age_list.append(pred_age)

                # Get correct number
                test_age_correct += (pred_age == y_age).sum().item()
                test_total_loss += loss_age.item()

            elif task == 'multi':
                # Get label
                y_gender = batch.gender.to(device)
                y_age = batch.age.to(device)
                y_total = torch.cat([y_gender, y_age])
                y_gender_list.append(y_gender)
                y_age_list.append(y_age)
                y_total_list.append(y_total)

                # Forward pass
                y_gender_pred, y_age_pred = model(text, sentiment_token, topic_token)

                # Get loss
                loss_gender = criterion(y_gender_pred, y_gender)
                loss_age = criterion(y_age_pred, y_age)
                loss = LAMBDA1 * loss_gender + LAMBDA2 * loss_age

                # Get prediction
                pred_gender = torch.argmax(y_gender_pred.data, dim=1)
                pred_age = torch.argmax(y_age_pred.data, dim=1)
                pred_total = torch.cat([pred_gender, pred_age])
                pred_gender_list.append(pred_gender)
                pred_age_list.append(pred_age)
                pred_total_list.append(pred_total)

                # Get correct number
                test_gender_correct += (pred_gender == y_gender).sum().item()
                test_age_correct += (pred_age == y_age).sum().item()
                test_total_correct += (pred_total == y_total).sum().item()
                test_total_loss += loss.item()

    print("##### Evaluation results #####")
    if task == 'gender':
        # 1. Concat tensors in a list
        # y_label
        y_gender_ = torch.cat(y_gender_list)
        # y_pred
        pred_gender_ = torch.cat(pred_gender_list)

        # 2. Statistics result
        # 1) Avg gender loss
        avg_loss = test_total_loss / len(test_data)
        print("Test gender avg loss: {}".format(avg_loss))

        # 2) Accuracy
        test_gender_acc = test_gender_correct / len(test_data)
        print("Test gender accuracy: {}".format(test_gender_acc))

        # 3) F1-score, (y_label, y_pred)
        test_gender_macrof1 = f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(),
                                       average='macro')
        print("Test gender F1-macro: {}".format(test_gender_macrof1))
        return avg_loss, test_gender_acc, test_gender_macrof1

    elif task == 'age':
        # 1. Concat tensors in a list
        # y_label
        y_age_ = torch.cat(y_age_list)
        # y_pred
        pred_age_ = torch.cat(pred_age_list)

        # 2. Statistics result
        # 1) Avg age loss
        avg_loss = test_total_loss / len(test_data)
        print("Test age avg loss: {}".format(avg_loss))

        # 2) Accuracy
        test_age_acc = test_age_correct / len(test_data)
        print("Test age accuracy: {}".format(test_age_acc))

        # 3) F1-score, (y_label, y_pred)
        test_age_macrof1 = f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro')
        print("Test age F1-macro: {}".format(test_age_macrof1))
        return avg_loss, test_age_acc, test_age_macrof1

    elif task == 'multi':
        # 1. Concat tensors in a list
        # 1) y_label
        y_gender_ = torch.cat(y_gender_list)
        y_age_ = torch.cat(y_age_list)
        y_total_ = torch.cat(y_total_list)

        # 2) y_pred
        pred_gender_ = torch.cat(pred_gender_list)
        pred_age_ = torch.cat(pred_age_list)
        pred_total_ = torch.cat(pred_total_list)

        # 2. Statistics result
        # 1) Calculate average loss
        avg_loss = test_total_loss / (2 * len(test_data))
        print("Test Average Loss: {}".format(avg_loss))

        # 2) Accuracy
        test_gender_acc = test_gender_correct / len(test_data)
        test_age_acc = test_age_correct / len(test_data)
        test_total_acc = test_total_correct / (2 * len(test_data))
        print("Test gender accuracy: {}".format(test_gender_acc))
        print("Test age accuracy: {}".format(test_age_acc))
        print("Test total accuracy: {}\n".format(test_total_acc))

        # 3) F1-score, (y_label, y_pred)
        test_gender_macrof1 = f1_score(y_gender_.detach().cpu().numpy(), pred_gender_.detach().cpu().numpy(),
                                       average='macro')
        print("Test gender F1-macro: {}".format(test_gender_macrof1))

        test_age_macrof1 = f1_score(y_age_.detach().cpu().numpy(), pred_age_.detach().cpu().numpy(), average='macro')
        print("Test age F1-macro: {}".format(test_age_macrof1))

        test_total_macrof1 = f1_score(y_total_.detach().cpu().numpy(), pred_total_.detach().cpu().numpy(),
                                      average='macro')
        print("Test total F1-macro: {}".format(test_total_macrof1))

        return avg_loss, test_gender_acc, test_age_acc, test_total_acc, test_gender_macrof1, test_age_macrof1, \
               test_total_macrof1


# Evaluation
if args.task == 'gender':
    # 1. Evaluation
    avg_loss, test_gender_acc, test_gender_macrof1 = evaluate(model=model, task=args.task)

    # 2. Save the test avg loss & acc
    save_test_result(filename=res_filename, task=args.task, avg_loss=avg_loss, test_gender_acc=test_gender_acc,
                     test_age_acc=None, test_total_acc=None, test_gender_macrof1=test_gender_macrof1,
                     test_age_macrof1=None, test_total_macrof1=None)
    print("Save results to txt file successfully!\n")

elif args.task == 'age':
    # 1. Evaluation
    avg_loss, test_age_acc, test_age_macrof1 = evaluate(model=model, task=args.task)

    # 2. Save the test avg loss & acc
    save_test_result(filename=res_filename, task=args.task, avg_loss=avg_loss, test_gender_acc=None,
                     test_age_acc=test_age_acc, test_total_acc=None, test_gender_macrof1=None,
                     test_age_macrof1=test_age_macrof1, test_total_macrof1=None)
    print("Save results to txt file successfully!\n")

elif args.task == 'multi':
    # 1. Evaluation
    avg_loss, test_gender_acc, test_age_acc, test_total_acc, test_gender_macrof1, test_age_macrof1, test_total_macrof1 \
        = evaluate(model=model, task=args.task)

    # 2. Save the test avg loss & acc
    save_test_result(filename=res_filename, task=args.task, avg_loss=avg_loss, test_gender_acc=test_gender_acc,
                     test_age_acc=test_age_acc, test_total_acc=test_total_acc, test_gender_macrof1=test_gender_macrof1,
                     test_age_macrof1=test_age_macrof1, test_total_macrof1=test_total_macrof1)
    print("Save results to txt file successfully!\n")

