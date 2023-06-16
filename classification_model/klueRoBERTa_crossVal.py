import pandas as pd
import numpy as np
import torch
import os
import random
from sklearn.model_selection import StratifiedKFold, train_test_split

from transformers import AutoModel, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

# for graphing
import matplotlib.pyplot as plt

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

CFG = {
    'EPOCHS': 20,
    'LEARNING_RATE':5e-5,
    'BATCH_SIZE':64,
    'SEED':40,
    'MAX_LENGTH': 65,
    'PATIENCE': 5,
    'K': 3,
    'WARMUP_RATIO': 0.1
}

#os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sent_type = {0:'angry', 1:'fear', 2:'happy', 3:'sad'}

class sentenceDataset(Dataset):
    def __init__(self, dataframe, sent_col, tokenizer, labels = None):
        texts = dataframe[sent_col].values.tolist()

        self.texts = [tokenizer(text, 
                                padding = 'max_length', 
                                max_length = CFG['MAX_LENGTH'], 
                                truncation = True, 
                                return_tensors='pt') 
                      for text in texts]
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
        klue_out = self.klue(input_ids= input_ids, attention_mask = attention_mask)[0][:,0]

        x = self.fc1(klue_out).to(device)
        x = self.softmax(x).to(device)

        return x

def calc_label_accuracy(X,Y):
    _, max_indices = torch.max(X, 1)
    label = [0 for _ in range(4)] #실제 dataset의 각 감정 레이블당 갯수 
    acc = [0 for _ in range(4)] #inference 후, 각 감정 레이블당 갯수 

    for i, j in zip(max_indices, Y):
        l_val = ((j == 1).nonzero().flatten().tolist())[0]
        label[l_val] += 1
        if i == l_val:
            acc[i] += 1

    ans = [] #각 레이블당 accuracy 
    for i in range(4):
      if label[i] != 0:
        ans.append(np.round(acc[i]/label[i], 2))
      else:
        ans.append(0)

    return ans

#plot for train loss and validation loss 
def loss_graph(train_loss, val_loss, file_path):
    fig, (ax1, ax2) = plt.subplots(ncols = 2, figsize = (13, 6))
    ax1.plot(train_loss)
    ax1.set_title('train loss')

    ax2.plot(val_loss)
    ax2.set_title('val loss')
                
    plt.savefig(file_path)
                
def train(model, train_dataloader, val_dataloader, learning_rate, epochs):
    avg_val_losses, avg_train_losses = [], [] #epoch 당 평균 loss

    criterion = {
        'type' : nn.CrossEntropyLoss().to(device)
        }

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
      {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
      {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
      ]

    optimizer = AdamW(optimizer_grouped_parameters, lr = learning_rate)  
    model = model.to(device)
    
    t_total = len(train_dataloader) * CFG['EPOCHS']
    warmup_step = int(t_total * CFG['WARMUP_RATIO'])

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

    for epoch in range(epochs):
        val_losses, train_losses = [], []
        
        train_accuracy = [0 for _ in range(4)]
        val_accuracy = [0 for _ in range(4)]
        
        model.train() 
        
        for batch_id, (train_input, type_label) in tqdm(enumerate(train_dataloader)):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)
            type_label = type_label.to(device)
            optimizer.zero_grad()
            
            type_output = model(input_ids, attention_mask) 

            loss = criterion['type'](type_output, type_label.float()) 
            loss.backward()
            optimizer.step()
            scheduler.step() 
            train_losses.append(loss.item())


            for i in range(4):
              acc = calc_label_accuracy(type_output, type_label)
              train_accuracy[i] += acc[i]
        
        
        for i in range(4):
          print(f"train acc for label {sent_type[i]}: {train_accuracy[i] / (batch_id+1)}")

        with torch.no_grad(): 
            model.eval() 
            
            # same process as the above
            for  batch_id, (val_input, vtype_label) in tqdm(enumerate(val_dataloader)):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)
                vtype_label = vtype_label.to(device)
                
                vtype_output = model(input_ids, attention_mask) 
                
                loss = criterion['type'](vtype_output, vtype_label.float()) 
                val_losses.append(loss.item())

                for i in range(4):
                  acc = calc_label_accuracy(vtype_output, vtype_label)
                  val_accuracy[i] += acc[i]
          
            for i in range(4):
              print(f"val acc for label {sent_type[i]}: {val_accuracy[i] / (batch_id+1)}")
            
            train_loss = np.average(train_losses)
            val_loss = np.average(val_losses)
            avg_train_losses.append(train_loss)
            avg_val_losses.append(val_loss)
            
            print(f"epoch {epoch+1} train_loss: {train_loss:.4f} val_loss: {val_loss:.4f}")

    loss_graph(avg_train_losses, avg_val_losses, 
               '/home/ubuntu/model/pre-trained/klueroberta_CrossVal.png')    
  
    return model  
            
def encode_label(data, label_col): 
    print(f"encode_label success")
    data_tmp =pd.get_dummies(data, columns=[label_col])
    data_labels = {
        'type': data_tmp.iloc[:, 1:5].values.tolist() #we have 4 labels 
        }
    return data_labels
    

def run_classifier():
    data_path = "/home/ubuntu/data/train_data/0209_final_augEditedData.csv"
    sent_col = '문장'
    label_col = 'new_label'
    model_name = 'klue/roberta-small'
    seed_everything(CFG['SEED']) # Seed 고정 
    
    data = pd.read_csv(data_path)
    train, test = train_test_split(data, test_size=0.2, random_state=CFG['SEED'])
    train = train.reset_index(drop=True)
    
    base_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = sentenceClassifier(base_model)
    #model.load_state_dict(torch.load('/home/ubuntu/model/pre-trained/klueroberta_crossVal_epoch40.pt', map_location = device))

    k = CFG['K']
    skf = StratifiedKFold(n_splits = k, shuffle = True, random_state=CFG['SEED'])
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train[sent_col], train[label_col])):
        print('Fold {}'.format(fold + 1))
        if fold == 0:
            train_labels = encode_label(train.loc[train_idx], label_col)
            val_labels = encode_label(train.loc[val_idx], label_col)
            
            train_sample = sentenceDataset(train.loc[train_idx], sent_col, tokenizer, train_labels)
            val_sample = sentenceDataset(train.loc[val_idx], sent_col, tokenizer, val_labels)
            
            train_dataloader = DataLoader(train_sample, batch_size=CFG['BATCH_SIZE'], 
                                          shuffle=True, num_workers = 0) 
            val_dataloader = DataLoader(val_sample, batch_size=CFG['BATCH_SIZE'], 
                                        shuffle = True, num_workers = 0)
    
            model = train(model, train_dataloader, val_dataloader, CFG['LEARNING_RATE'], CFG['EPOCHS'])
            
    #evaluation
    test_labels = encode_label(test, label_col)
    test_accuracy = [0 for _ in range(4)]
    
    test_set = sentenceDataset(test, sent_col, tokenizer, test_labels)
    test_dataloader = DataLoader(test_set, batch_size=CFG['BATCH_SIZE'],
                                 shuffle = True, num_workers = 0)
    
    with torch.no_grad(): 
        model.eval()
        for batch_id, (test_input, t_label) in tqdm(enumerate(test_dataloader)):
            attention_mask = test_input['attention_mask'].to(device)
            input_ids = test_input['input_ids'].squeeze(1).to(device)
            t_label = t_label.to(device)
            
            test_output = model(input_ids, attention_mask) # from the forward function

            for i in range(4):
                acc = calc_label_accuracy(test_output, t_label)
                test_accuracy[i] += acc[i]
        
        for i in range(4):
            print(f"test acc for label {sent_type[i]}: {test_accuracy[i] / (batch_id+1)}")
