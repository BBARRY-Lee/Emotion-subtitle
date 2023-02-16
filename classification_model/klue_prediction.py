import pandas as pd
import numpy as np
import torch
import os
import random

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch import nn
from tqdm import tqdm
from sklearn.pipeline import make_pipeline

CFG = {
    'BATCH_SIZE':64,
    'SEED':42,
    'MAX_LENGTH': 65,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sent_type = {0: 'angry', 1: 'fear', 2:'happy', 3:'sad'}

class text_preprocess():
  def __init__(self, sent_col):
    self.sent_col = sent_col

  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    new_X = X[['Start Time', 'End Time', self.sent_col]].copy() 
    new_X[self.sent_col] = X[self.sent_col].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9 ?!]', '', regex=True) 
    new_X['Start Time'] = new_X['Start Time'].astype(str)
    new_X['End Time'] = new_X['End Time'].astype(str)
    return new_X

class sentenceDataset(Dataset):
    def __init__(self, dataframe, sent_col, tokenizer, labels=None):
        texts = dataframe[sent_col].values.tolist()

        self.texts = [tokenizer(text, padding='max_length', max_length = CFG['MAX_LENGTH'], truncation=True, return_tensors='pt') for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        if self.labels is not None:
            type_tmp = self.labels['type'][idx]
            return text, torch.Tensor(type_tmp).to(device)
        else:
            return text, torch.Tensor([-1,-1,-1,-1]).to(device)
        
class sentenceClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.klue = base_model # from transformers package

        self.fc1 = nn.Linear(768, 4).to(device)
        self.softmax = nn.Softmax(dim = 1).to(device)

    def forward(self, input_ids, attention_mask):
        # input_ids : token's id / attention_mask : make a model to focus on which token
        klue_out = self.klue(input_ids = input_ids, attention_mask = attention_mask)[0][:,0]

        x = self.fc1(klue_out).to(device)
        x = self.softmax(x).to(device)

        return x
  
class kluePredictor():
  def __init__(self, model_path, model_name, device, sent_col): 
    self.model_name = model_name
    self.device = device
    self.base_model = AutoModel.from_pretrained(model_name).to(device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.sent_col = sent_col
    self.model = sentenceClassifier(self.base_model)
    self.model.load_state_dict(torch.load(model_path))
    print("model load success!")
  
  def fit(self):
    return self

  def transform(self, input_data):
    test_data = sentenceDataset(input_data, self.sent_col, self.tokenizer)
    test_dataloader = DataLoader(test_data, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers = 0) 
    
    self.model.eval()

    result = []
    with torch.no_grad(): 
        for batch_id, (test_input, t_label) in tqdm(enumerate(test_dataloader)):
            attention_mask = test_input['attention_mask'].to(device)
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            t_label = t_label.to(device)
            
            test_output = self.model(input_ids, attention_mask) # from the forward function

            for j in test_output: 
              vec = j.detach().cpu().numpy()
              if max(vec) < 0.3:
                result.append(4)
              else:
                label = np.where(vec == max(vec))[0]
                result.append(label[0])
           
    return pd.concat((input_data[['Start Time','End Time', 'Text']], pd.DataFrame({'Label': result})), axis = 1)


def eval(input_path, model_path, model_name, sent_col):
  #test_input = pd.read_csv(input_path, encoding = 'cp949')
  test_input = pd.read_csv(input_path)
  pipe = make_pipeline(text_preprocess(sent_col = sent_col), 
                       kluePredictor(model_path, model_name, 
                                     device = device,
                                     sent_col = sent_col))
  labeled_script = pipe.transform(test_input)
  labeled_script.to_csv("/home/ubuntu/myapp/composite_code/input_text/script_test_result.csv", index = False, encoding = 'utf-8')

