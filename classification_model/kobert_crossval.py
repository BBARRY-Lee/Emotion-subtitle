#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, SubsetRandomSampler


import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model
from transformers import AdamW

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair, mode = "train"):
        self.mode = mode
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, pad = pad, pair = pair)
        if self.mode == "train":
            self.sentences = [transform([i[sent_idx]]) for i in dataset]
            self.labels = [np.int32(i[label_idx]) for i in dataset]          
        else:
            self.sentences = [transform(i) for i in dataset]
        
    def __getitem__(self, i):
        if self.mode == 'train':
            return (self.sentences[i] + (self.labels[i], ))
        else:
            return self.sentences[i]
    
    def __len__(self):
        return (len(self.sentences))


class BERTClassifier(nn.Module):
    def __init__(self, bert, hidden_size = 768, num_classes=4, dr_rate=None, params=None): #클래스 수 조정
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


def calc_label_accuracy(X,Y):
    _, max_indices = torch.max(X, 1)
    label = [torch.numel(Y[Y==i]) for i in range(4)] #실제 dataset의 각 레이블당 갯수 
    acc = [0 for _ in range(4)]
    for i, j in zip(max_indices, Y):
      if i == j:
        acc[i] += 1
    
    ans = []
    for i in range(4):
      if label[i] != 0:
        ans.append(acc[i]/label[i])
      else:
        ans.append(0)

    return ans

def run_classifier():
  data_path = "/home/ubuntu/data/train_data/0209_final_augEditedData.csv"
  sent_col = '문장'
  label_col = 'new_label'
  
  txt_data = pd.read_csv(data_path, encoding = 'utf-8')
  txt_data[sent_col] = txt_data[sent_col].astype(str) 
  
  le = LabelEncoder()
  le.fit(txt_data[label_col].unique())
  txt_data['le_sentiment'] = le.transform(txt_data[label_col])
  
  bertmodel, vocab = get_pytorch_kobert_model(cachedir = ".cache")
  tokenizer = get_tokenizer()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower = False)

  max_len = 50 
  batch_size = 64
  num_epochs = 15
  max_grad_norm = 1
  learning_rate = 5e-5

  train, test = train_test_split(txt_data.drop('new_label', axis = 1), test_size=0.3, shuffle = True, random_state=42, stratify= txt_data['le_sentiment'])

  label_idx, sent_idx = 1, 0
  data_train = BERTDataset(train.values, sent_idx, label_idx, tok, max_len, True, False, mode = "train")
  data_test = BERTDataset(test.values, sent_idx, label_idx, tok, max_len, True, False, mode = "train")

  k = 5
  splits = KFold(n_splits=k,shuffle=True,random_state=42)
  
  model = BERTClassifier(bertmodel, dr_rate = 0.5).to(device) #overfitting 경향성 dropout 비율 높임

  no_decay = ['bias', 'LayerNorm.weight']
  optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
  ]

  optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
  loss_fn = nn.CrossEntropyLoss()
    
  for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(train)))):
    print('Fold {}'.format(fold + 1))

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5, sampler = train_sampler)
    val_dataloader = torch.utils.data.DataLoader(data_train, batch_size = batch_size, num_workers = 5, sampler = val_sampler)

    avg_val_losses, avg_train_losses = [], [] #epoch 당 평균 loss
    
    for e in range(num_epochs):
      val_losses, train_losses = [], []
      
      train_accuracy = [0 for _ in range(4)]
      val_accuracy = [0 for _ in range(4)]
      
      model.train()
      for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
          for param in model.parameters():
            param.grad = None
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          label = label.long().to(device)
          out = model(token_ids, valid_length, segment_ids)
          loss = loss_fn(out, label)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          optimizer.step()
          train_losses.append(loss.item())

          for i in range(4):
            acc = calc_label_accuracy(out, label)
            train_accuracy[i] += acc[i]

      for i in range(4):
        print(f"train acc for label {le.inverse_transform([i])[0]}: {train_accuracy[i] / (batch_id+1)}")
      
      model.eval()
      with torch.no_grad():
        for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids .long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = model(token_ids, valid_length, segment_ids) #out: 각 class에 대한 softmax값 
            loss = loss_fn(out, label)
            val_losses.append(loss.item())
            
            for i in range(4):
              acc = calc_label_accuracy(out, label)
              val_accuracy[i] += acc[i]
        
        for i in range(4):
          print(f"val acc for label {le.inverse_transform([i])[0]}: {val_accuracy[i] / (batch_id+1)}")
          
        train_loss = np.average(train_losses)
        val_loss = np.average(val_losses)
        avg_train_losses.append(train_loss)
        avg_val_losses.append(val_loss)
        
        print(f"epoch {e+1} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")
        
  model.eval()
  test_accuracy = [0 for _ in range(4)]
  test_dataloader = torch.utils.data.DataLoader(data_test, batch_size = batch_size, num_workers = 5)

  with torch.no_grad():
      for batch_id, (token_ids, valid_length, segment_ids, label) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          label = label.long().to(device)
          out = model(token_ids, valid_length, segment_ids)

          for i in range(4):
            acc = calc_label_accuracy(out, label)
            test_accuracy[i] += acc[i]

  for i in range(4):
    print(f"test acc for label {le.inverse_transform([i])[0]} : {test_accuracy[i] / (batch_id+1)}")