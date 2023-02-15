import pandas as pd
from glob import glob
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from sklearn.pipeline import make_pipeline

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#하이퍼 파라미터 세팅
max_len = 80
batch_size = 64
warmup_ratio = 0.1
#num_epochs = 1
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

class text_preprocess():

  def fit(self, X, y=None):
    return self
  
  def transform(self, X):
    new_X = X.copy()
    new_X['Text'] = X['Text'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 0-9 ?!]', '', regex=True) 

    return new_X

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                pad, pair, mode = "train"):
        self.mode = mode
        transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length = max_len, 
                                                   pad = pad, pair = pair)
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
      
      _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), 
                            attention_mask = attention_mask.float().to(token_ids.device))
      if self.dr_rate:
          out = self.dropout(pooler)
      return self.classifier(out)
    
    

class BERTPredictor():
  def __init__(self, model_path, device): 
    self.bertmodel, self.vocab = get_pytorch_kobert_model(cachedir = ".cache")
    self.tokenizer = get_tokenizer()
    self.tok = nlp.data.BERTSPTokenizer(self.tokenizer, self.vocab, lower = False)
    self.device = device
    self.model = BERTClassifier(self.bertmodel, dr_rate = 0.3).to(device)
    self.model.load_state_dict(torch.load(model_path, map_location = self.device))
    print("model load success!")
  
  def fit(self):
    return self

  def transform(self, input_data, sent_idx = 2):

    test_data = BERTDataset(input_data['Text'], sent_idx, 1, 
                            self.tok, max_len, True, False, mode = 'test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                                  num_workers=5)
    
    self.model.eval()

    result = []
    output_vec = []
    m = nn.Softmax(dim= -1)
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(self.device)
        segment_ids = segment_ids.long().to(self.device)
        valid_length = valid_length
        out = self.model(token_ids, valid_length, segment_ids)#.detach().cpu().numpy()
        for j in out:
          prob = []
          output = m(j)
          output_vec.append(output.detach().cpu().numpy())
          neu_flag = True
          for val in output.detach().cpu().numpy():
            if val >= 0.3:
              neu_flag = False

          label = 4 if neu_flag else int(torch.argmax(output))
          result.append(label)
          
        print(result)

    return pd.concat((input_data, pd.DataFrame({'label': result})), axis = 1)

def eval(input_path, model_path):
  print(input_path)
  test_input = pd.read_csv(input_path, encoding='cp949')
  print(test_input)
  model_path = model_path
  pipe = make_pipeline(text_preprocess(), 
                       BERTPredictor(model_path = model_path, 
                                     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')))
  labeled_script = pipe.transform(test_input)
  labeled_script.to_csv("/home/ubuntu/data/test_result.csv", index = False, encoding = 'utf-8-sig')
    
