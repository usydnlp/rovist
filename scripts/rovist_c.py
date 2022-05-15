from transformers import AlbertTokenizer, AlbertModel
import argparse
import pandas as pd 
import os
import datetime
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('We are using GPU.' if use_cuda else 'We are using CPU.')

model_nm = 'albert-large-v1'
tokenizer = AlbertTokenizer.from_pretrained(model_nm)

def test_loader(dl, index): 
    """
    Sanity check the dataloader. 
    """
    print('Token tensors for this sample:', dl[index]['tokens_tensor'])
    print('Segment IDs for this sample:', dl[index]['segments_tensor'])
    print('The class this sample: ', dl[index]['target'])

class ALBERTDataLoader(Dataset):
    def __init__(self, dataframe): 

        self.dataframe = dataframe
        self.prev_sents = dataframe['Sent A'].to_list()
        self.next_sents = dataframe['Sent B'].to_list()
        self.targets = dataframe['Class'].to_list()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx): 
    
        sent1 = self.prev_sents[idx]
        sent2 = self.next_sents[idx]

        # get sentence lengths 
        sent1_len = len(tokenizer.tokenize(sent1))
        sent2_len = len(tokenizer.tokenize(sent2))

        target = self.targets[idx]

        return sent1, sent2, sent1_len, sent2_len, target

def collate_fn(data):
    """
    Collate data for each mini-batch.
    Data comes as a list of tuples (prev_sent, next_sent, prev_sent_len, 
    next_sent_len, sentence order label)

    Returns: token ids, sequence ids, attention masks and labels.
    """
    # sort a data list by sequence lenngth
    data.sort(key=lambda x: x[2]+x[3], reverse=True)
    sent1, sent2, sent1_len, sent2_len, targets = zip(*data)

    max_len = sent1_len[0] + sent2_len[0]
    tokens_tensor, segments_tensor, attentions_tensor = [], [], []
    sent_order_labels = []

    for i in range(len(data)): 

        sentA = sent1[i]
        sentB = sent2[i]

        # length of the sequence 
        seq_len = sent1_len[i] + sent2_len[i]

        pad_length = max_len - seq_len
        text_list = ['[CLS]', sentA, '[SEP]', sentB, '[SEP]']

        text = ' '.join(text_list)
            
        # Tokenize input
        tokenized_text = tokenizer.tokenize(text)
        tokenized_text.extend(['[PAD]'] * pad_length)

        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        # 0 for the previous sentence, 1 for the following sentence 
        segments_ids = (sent1_len[i]+2)*[0] + (sent2_len[i]+1)*[1] + (pad_length)*[1]

        # mask the padding tokens. 0 means to mask the token
        attention_ids = (sent1_len[i]+2)*[1] + (sent2_len[i]+1)*[1] + (pad_length)*[0]

        tokens_tensor.append(torch.tensor([indexed_tokens]))
        segments_tensor.append(torch.tensor([segments_ids]))
        attentions_tensor.append(torch.tensor([attention_ids]))

    # Convert inputs to PyTorch tensors and move to device 
    tokens_tensor = torch.stack(tokens_tensor).squeeze(1).to(device)
    segments_tensor = torch.stack(segments_tensor).squeeze(1).to(device)
    attentions_tensor = torch.stack(attentions_tensor).squeeze(1).to(device)

    targets = np.asarray(targets)
    targets = torch.from_numpy(targets).to(device)

    return tokens_tensor, segments_tensor, attentions_tensor, targets

class SOPClassifier(nn.Module):
    def __init__(self, hidden_dim, dropout_prob, model_nm):
        """
        Albert Model with head for SOP. 
        """
        super(SOPClassifier, self).__init__()

        self.albert_model = AlbertModel.from_pretrained(model_nm)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, tokens_tensor, attentions_tensor, segments_tensor): 
        
        # pooled_seq shape = (batch_size, 1024)
        pooled_seq = self.albert_model(tokens_tensor, 
                                    attention_mask = attentions_tensor, 
                                    token_type_ids = segments_tensor)[1]

        # SOP classifier. Outputs are logits 
        pooled_output = self.dropout(pooled_seq)
        output = self.linear(pooled_seq)

        return output

def train_epoch(model, dataloader, criterion, optimizer, mode = 'train'): 
    """
    Function to train each epoch. 

    Args: 
        dataloader: either train or valid dataloader. 
        criterion: loss function to use. 
        optimizer: optimizer for training. 
        mode: either "train" or "validate". 
    """
    time1 = datetime.datetime.now()

    if mode == 'train': 
        model.train()
    else: #put model in validation mode 
        model.eval()
        
    #keep track of training and validation loss and accuracy 
    running_loss, running_acc = 0, 0 

    # mini-batch training with the dataloader 
    for batch_idx, data in enumerate(dataloader): 
       
        # move this batch of data to specified device 
        tokens_tensor, segments_tensor, attentions_tensor, targets = data

        # gradient calculation when training
        with torch.set_grad_enabled(mode =="train"):
            
            # forward data through model to get logits 
            outputs = model(tokens_tensor, 
                            attentions_tensor, 
                            segments_tensor)

            loss = criterion(outputs, targets)

            probs = F.log_softmax(outputs, dim=1)
            predictions = torch.argmax(probs, dim=1)
            # move targets back to CPU for comparison
            predictions = predictions.to('cpu').to(torch.int).numpy()
            targets = targets.to('cpu').to(torch.int).numpy()

            acc = accuracy_score(predictions, targets)

            if mode == 'train': 
                loss.backward()       # backward the loss and calculate gradients for parameters.
                optimizer.step()      # update the parameters.
                optimizer.zero_grad() # zero the gradient to stop from accumulating
        
        if (batch_idx + 1) % 100 == 0: 
            print("Processed batch: {}. Loss: {}. Acc: {}.".format(batch_idx+1, loss.item(), acc))

        running_loss += loss.item()
        running_acc += acc # mean accuracy of all examples in batch 

    # note len(dataloader) is number of batches
    epoch_loss = running_loss/len(dataloader) # len(dataloader) = no. of examples / batch size 
    epoch_acc = running_acc/len(dataloader) # mean accuracy of all batches
    time2 = datetime.datetime.now()

    return epoch_loss, epoch_acc, (time2-time1).total_seconds()

def train_model(model, training_info, opt, start_epoch = 0): 
    """
    Function for model training. 

    Args:
        model: initialised pytorch model (class)
        training_info: dict of loader, criterion and optimizer information. 
        opt: the parsed arguments. 
        start_epoch: starting epoch. Will be >0 if loaded model checkpoint. 
    """

    MIN_LOSS = float('inf')
    EARLY_STOPPING_COUNT = 0
    EVAL_EVERY_EPOCH = 1
    
    scheduler = training_info["scheduler"]

    for epoch in range(start_epoch, training_info["num_epochs"]): 

        # forward training data through model
        train_loss, train_acc, runtime = train_epoch(model, 
                                                    training_info["train_loader"], 
                                                    training_info["criterion"],
                                                    training_info["optimizer"],
                                                    mode = 'train')

        print("Epoch:%d, train loss: %.4f, train acc: %.4f, time: %.2fs" %(epoch+1, train_loss, train_acc, runtime))
        
        if (epoch + 1) % EVAL_EVERY_EPOCH == 0: 
            valid_loss, valid_acc, runtime = train_epoch(model, 
                                                         training_info["valid_loader"], 
                                                         training_info["criterion"],
                                                         training_info["optimizer"],
                                                         mode = 'validate')
                
            print('-'*60)
            print("Epoch:%d, valid loss: %.4f, valid acc: %.4f, time: %.2fs" %(epoch+1, valid_loss, valid_acc, runtime))
            print('-'*60)

            """
            CHECK EARLY STOPPING CONDITIONS
            """
            if valid_loss < MIN_LOSS:
                MIN_LOSS = valid_loss
                EARLY_STOPPING_COUNT = 0

                # save the best model so far 
                state = {"epoch": epoch, "model": model.state_dict(), "valid_loss": valid_loss, 
                        "valid_acc": valid_acc, "train_loss": train_loss, "train_acc": train_acc,
                        "opt": opt}
                model_name = "sop_model_epoch{}.pth.tar".format(epoch)
                torch.save(state, os.path.join(training_info["save_path"], model_name))
            else: 
                EARLY_STOPPING_COUNT += 1
            
            if EARLY_STOPPING_COUNT == training_info["num_es_epochs"]:
                break
            
            # apply learning rate decay
            scheduler.step() 

def main(): 
    
    ### HYPER-PARAMETERS 

    parser = argparse.ArgumentParser() 

    parser.add_argument("--data_path", 
                        default = "/content",
                        help = "path to training and validation data.")
    parser.add_argument("--batch_size", default = 32, type = int, 
                        help = "size of training mini-batch.")
    parser.add_argument("--learning_rate", default = 0.00001, type = float,
                        help = "initial learning rate.")
    parser.add_argument("--dropout_prob", default = 0.40, type = float,
                        help = "Dropout probability to use before classifier.")
    parser.add_argument("--lr_decay", default = 0.95, type = float,
                        help = "gamma for learning rate scheduler.")
    parser.add_argument("--weight_decay", default = 1e-5, type = float,
                        help = "weight decay for optimizer.")
    parser.add_argument("--num_epochs", default = 50, type = int,
                        help = "number of training epochs.")
    parser.add_argument("--hidden_dim", default = 1024, type = int,
                        help = "hidden dimension of linear layer on top of ALBERT")
    parser.add_argument("--early_stop", default = 5, type = int, 
                        help = "number of epochs used for early stopping.")
    parser.add_argument("--save_path", 
                        default = "/content", 
                        help = "path to where the model weights are saved.")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    opt = parser.parse_args()
    print(opt)

    ### load data from path 
    train_data = pd.read_csv(os.path.join(opt.data_path, "albert_train.csv"))
    valid_data = pd.read_csv(os.path.join(opt.data_path, "albert_valid.csv"))

    ### create data loaders for training and validation set 

    train_dl = ALBERTDataLoader(train_data)
    valid_dl = ALBERTDataLoader(valid_data)
    train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, 
                                                batch_size = opt.batch_size,
                                                collate_fn = collate_fn)

    valid_dataloader = torch.utils.data.DataLoader(valid_dl, shuffle = True, 
                                                batch_size = opt.batch_size,
                                                collate_fn = collate_fn)
    

    ### initialise model and gather training info

    model = SOPClassifier(opt.hidden_dim, opt.dropout_prob, model_nm)
    model = model.to(device)

    ### optimizer and loss functions
    criterion = nn.CrossEntropyLoss() #binary cross entropy loss 
    optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate, 
                           weight_decay = opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1.0, gamma=opt.lr_decay)

    training_info  = {"num_epochs": opt.num_epochs,
                        "criterion": criterion, 
                        "optimizer": optimizer,
                        "scheduler": scheduler, 
                        "num_es_epochs": opt.early_stop,  
                        "train_loader": train_dataloader,
                        "valid_loader": valid_dataloader, 
                        "save_path": opt.save_path}

    if opt.resume: 
        ### start training model from loaded check point 
        if os.path.isfile(opt.resume):
            checkpoint = torch.load(opt.resume)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint["epoch"] + 1 

            print("Loaded model checkpoint! Starting at epoch: {}".format(start_epoch))

            train_model(model, training_info, opt, start_epoch = start_epoch)
    else: 
        ### train model from scratch 
        train_model(model, training_info, opt)

if __name__ == '__main__':
    main()