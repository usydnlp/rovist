import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import pandas as pd 
import numpy as np
import json
import argparse
import os 
import datetime
from PIL import Image

# from transformers import BertTokenizer, BertModel
from transformers import ViTFeatureExtractor, ViTModel

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

import nltk
nltk.download('punkt')

# Enable GPU
use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We are using GPU.' if use_cuda else 'We are using CPU.')

def init_glove_model(path, word_dim = 300): 
    """
    Initialises the GloVE Embedding model. Text embeddings are 300 dimension.
    """
    glove_input_file = 'glove.6B.{}d.txt'.format(word_dim)
    word2vec_output_file = os.path.join(path, 'glove.6B.{}d.txt.word2vec'.format(word_dim))
    glove2word2vec(glove_input_file, word2vec_output_file)

    model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)

    return model 


def collate_fn_contextualized(data):
    """
    Data comes in as list of tuples (tensor images, tokenized phrase)
    Use this to collate data if using BERT embeddings. 
    """
    # sort a data list by sequence lenngth
    data.sort(key=lambda x: len(x[1]) , reverse=True)

    images, phrase_tokens = zip(*data)
    max_len = len(phrase_tokens[0])

    token_ids_tensors = [] 
    attention_ids_tensors = []

    for i in range(len(data)): 
        
        tokenized_phrase = phrase_tokens[i]
        phrase_len = len(tokenized_phrase)
        pad_len = max_len - phrase_len

        # tokenizer is a global variable 
        tokenized_phrase.extend(['[PAD]'] * pad_len)
        token_ids = tokenizer.convert_tokens_to_ids(tokenized_phrase)

        attention_ids = [1] * phrase_len + [0] * pad_len

        token_ids_tensors.append(torch.tensor(token_ids))
        attention_ids_tensors.append(torch.tensor(attention_ids))
    
    token_ids_tensors = torch.stack(token_ids_tensors) # (batch_size, max_len)
    attention_ids_tensors = torch.stack(attention_ids_tensors) # (batch_size, max_len)

    if type(images[0]) == torch.Tensor:
        images = torch.stack(list(images)) # (batch_size, 3, 224, 224)
    else: 
        images = list(images) # list of PIL images
    
    batch_data = {"images": images, 
                   "token_ids": token_ids_tensors, 
                   "attention_ids": attention_ids_tensors}

    return batch_data

def collate_fn_static(data): 
    """
    Data comes in as list of tuples (tensor images, tokenized phrase). 
    Use this to collate data if using glove embeddings. 
    """
    images, phrase_embs = zip(*data)

    if type(images[0]) == torch.Tensor:
        images = torch.stack(list(images)) # (batch_size, 3, 224, 224)
    else: 
        images = list(images) # list of PIL images
    
    batch_data = {"images": images, 
                   "phrase_embeddings": torch.stack(list(phrase_embs))}

    return batch_data

class Flickr30kEntsLoader(Dataset): 

    def __init__(self, dataframe, image_path, word_emb_model = None,
                 img_model_type = "vision transformer", word_emb_type = "static"): 
        """
        Args: 
            dataframe: Flickr30k bounding boxe with region phrase dataframe.
            image_path: path to Flickr30k bounding box images.   
            word_emb_model: glove model or None if using BERT model.
            image_model_type: either "resnet" or "vision transformer".
            word_emb_type: either "contextualised" or "static". 
        """

        self.dataframe = dataframe 
        self.image_ids = dataframe["image_id"].tolist() 
        self.noun_phrases = dataframe["noun_phrase"].tolist() 
        self.bboxes = dataframe["bbox"].tolist() 

        self.image_path = image_path

        self.preprocess = preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        
        self.img_model_type = img_model_type
        self.word_emb_type = word_emb_type

        self.word_emb_model = word_emb_model # glove model.

    def __len__(self): 

        return len(self.dataframe)

    def __getitem__(self, idx): 

        image_id = self.image_ids[idx]
        bbox = self.bboxes[idx]

        image_name = str(image_id) + "_" + bbox + ".jpg"
        img = Image.open(self.image_path + '/' + image_name).convert('RGB')

        # preprocess images if using resnet 
        if self.img_model_type == "resnet": 
            img = self.preprocess(img) # tensor of shape (3, 224, 224)

        phrase = self.noun_phrases[idx]

        if self.word_emb_type == "contextualized": # use bert tokenizer
            tokenized_phrase = self.tokenizer.tokenize(phrase)

            return img, tokenized_phrase

        # Use glove embeddings. Stop words should already be removed from Flickr30k Entities Dataset. 
        if self.word_emb_type == "static": 
            tokenized_phrase = nltk.word_tokenize(phrase)
            word_embs = []
            for word in tokenized_phrase: 
                try: 
                    word_embs.append(self.word_emb_model[word])
                except: 
                    word_embs.append(self.word_emb_model["unk"])
            
            phrase_embedding = torch.tensor(np.mean(word_embs, axis = 0))

            return img, phrase_embedding

class ImageEncoder(nn.Module): 

    def __init__(self, model_type, joint_emb_dim, activate_fn = "Tanh"): 
        """
        Encodes image to produce image embeddings. 

        Args: 
            model_type: either "resnet" for ResNet152 or "vision transformer" for VIT.
            joint_emb_dim: dimension of embeddings.   
            activate_fn: either tanh, leaky relu, sigmoid or relu. 
        """
        super(ImageEncoder, self).__init__()

        self.model_type = model_type

        if activate_fn == 'Tanh':
            self.activate_fn = nn.Tanh()
        elif activate_fn == 'Leaky ReLU':
            self.activate_fn = nn.LeakyReLU(0.1)
        elif activate_fn == 'Sigmoid': 
            self.activate_fn = nn.Sigmoid()
        else:
            self.activate_fn = nn.ReLU() 

        if self.model_type == "resnet": 
            print("Using ResNet152!")

            self.image_model = models.resnet152(pretrained = True)

            modules = list(self.image_model.children())[:-1]
            self.image_model = nn.Sequential(*modules)
            for p in self.image_model.parameters():
                p.requires_grad = False

            self.linear = nn.Linear(2048, joint_emb_dim)

        elif self.model_type == "vision transformer": 
            print("Using Vision Tranformer!")

            self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
            self.image_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

            self.linear = nn.Linear(768, joint_emb_dim)

    def forward(self, images):
        
        if self.model_type == "resnet": 
            # remember to move images to device 
            images = images.to(device)
            resnet_feats = self.image_model(images) # (batch_size, 2048, 1, 1) 
            output = self.linear(resnet_feats.view(-1, 2048))

        elif self.model_type == "vision transformer": 
      
            img_inputs = self.feature_extractor(images = images, 
                                                return_tensors = "pt")
            
            # remember to move images to device 
            img_inputs = img_inputs['pixel_values'].to(device)

            vit_feats = self.image_model(img_inputs)[1] # (batch_size, 768)
            output = self.linear(vit_feats) 

        output = self.activate_fn(output)  # (batch_size, joint_emb_dim)

        return output 

class TextEncoder(nn.Module): 

    def __init__(self, joint_emb_dim, word_emb_model = None, word_emb_type = "static", 
                 activate_fn = "Tanh", hidden_dim = 300): 
        super(TextEncoder, self).__init__()
        """
        Encodes text/noun phrase to produce text embeddings. 

        Args: 
            word_emb_model: None if using glove embeddings. 
            joint_emb_dim: dimension of word embeddings.   
            activate_fn: either tanh, leaky relu, sigmoid or relu. 
            hidden_dim: size of initial word embeddings. Will be 300 if using glove. 
            word_emb_type: either "contexualized" for BERT or "static" for glove.

        """

        self.word_emb_model = word_emb_model 
        self.word_emb_type = word_emb_type

        if word_emb_type == "contextualized": 

            self.linear = nn.Linear(1024, joint_emb_dim)

        elif word_emb_type == "static": 
            
            self.linear = nn.Linear(300, joint_emb_dim)

        if activate_fn == 'Tanh':
            self.activate_fn = nn.Tanh()
        elif activate_fn == 'Leaky ReLU':
            self.activate_fn = nn.LeakyReLU(0.1)
        elif activate_fn == 'Sigmoid': 
            self.activate_fn = nn.Sigmoid()
        else:
            self.activate_fn = nn.ReLU() 

    def forward(self, batch_data): 

        # pooled output is second element in tuple 
    
        if self.word_emb_type == "contextualized": # feed through BERT
            output = self.word_emb_model(batch_data["token_ids"].to(device), 
                                             batch_data["attention_ids"].to(device))[1] 
        elif self.word_emb_type == "static": 

            output = self.linear(batch_data["phrase_embeddings"].to(device)) 

        output = self.activate_fn(output) # (batch_size, joint_emb_dim)

        return output 

class SymmetricLoss(nn.Module):
    """
    Compute Symmetric loss.  
    """
    def __init__(self, gamma = 1.0):

        super(SymmetricLoss, self).__init__()
   
        self.temperature = Variable(torch.tensor(gamma), requires_grad=True)
        self.log_softmax = nn.LogSoftmax(dim = -1)

    def forward(self, image_embeddings, text_embeddings):

        # text to image sims: logits are not softmaxed (batch_size, batch_size)
        logits = (text_embeddings @ image_embeddings.T) * self.temperature

        # image and text similarity with itself. Should have 1s in the diagonals 
        images_sim = image_embeddings @ image_embeddings.T 
        texts_sim = text_embeddings @ text_embeddings.T 

        # softmaxed identity matrix. 
        labels = F.softmax((images_sim + texts_sim)/2 * self.temperature, 
                            dim = -1)

        images_loss = (-labels.T * self.log_softmax(logits.T)).sum(dim = 1)
        texts_loss = (-labels * self.log_softmax(logits)).sum(dim = 1)

        loss = (images_loss + texts_loss)/2 # (batch_size)
        loss = loss.mean()

        return loss 

class Region2PhraseModel(nn.Module):
    
    def __init__(self, joint_emb_dim, image_model_type = "vision transformer", 
                 word_emb_model = None, word_emb_type = "static", activate_fn = "Tanh"):
        super(Region2PhraseModel, self).__init__()
        """
        Combine entire model.  

        Args: 
            word_emb_model: None if using glove embeddings. 
            joint_emb_dim: dimension of word embeddings.   
            activate_fn: either tanh, leaky relu, sigmoid or relu. 
            image_model_type: either "resnet" for ResNet152 or "vision transformer" for VIT.
            word_emb_type: either "contexualized" for BERT or "static" for glove.
        """

        self.image_encoder = ImageEncoder(image_model_type, joint_emb_dim, 
                                          activate_fn = activate_fn)

        self.text_encoder = TextEncoder(joint_emb_dim, word_emb_model, word_emb_type,
                                        activate_fn = activate_fn)

    def l2norm(self, X, dim, eps = 1e-8):
        """
        L2-normalize columns of X
        """
        # sum across the rows 
        norm = torch.pow(X, 2).sum(dim = dim, keepdim = True).sqrt() + eps
        X = torch.div(X, norm)
        
        return X

    def forward(self, batch_data): 

        # returned embedding shape = (batch_size, joint_emb_size)

        image_feats = self.image_encoder(batch_data["images"])
        #image_embeddings = self.l2norm(image_feats, dim = -1)

        text_feats = self.text_encoder(batch_data)
        #text_embeddings = self.l2norm(text_feats, dim = -1)

        return image_feats, text_feats

def train_epoch(model, dataloader, criterion, optimizer, epoch, save_path, 
                mode = 'train'): 
    """
    Function to train each epoch. 

    Args: 
        dataloader: either train or valid dataloader. 
        criterion: loss function to use. 
        optimizer: optimizer for training. 
        mode: either "train" or "validate". 
        epoch: current epoch to process. 
        save_path: path to save the trained weights. 
    """
    time1 = datetime.datetime.now()

    if mode == 'train': 
        model.train()
    else: #put model in validation mode 
        model.eval()
        
    #keep track of training and validation loss and accuracy 
    running_loss = 0 

    # mini-batch training with the dataloader 
    for batch_idx, data in enumerate(dataloader): 
      
        batch_data = data
        
        with torch.set_grad_enabled(mode =="train"):
            
            # forward data through model to get logits 
            image_embeddings, text_embeddings = model(batch_data)
       
            # obtain symmetric loss 
            loss = criterion(image_embeddings, text_embeddings)

            if mode == 'train': 
                loss.backward()       
                optimizer.step()      
                optimizer.zero_grad()
        
        if (batch_idx + 1) % 1 == 0: 
            print("Processed batch: {}. Loss: {}.".format(batch_idx+1, loss.item()))

        # save progress every 1000 iterations. 
        # if (batch_idx + 1) % 1000 == 0 and mode == "train": 
        #     state = {"epoch": epoch, "model": model.state_dict(), "valid_loss": "None", 
        #             "train_loss": running_loss/(batch_idx + 1)}
        #     model_name = "r2p_epoch{}_batch{}.pth.tar".format(epoch, batch_idx)
        #     torch.save(state, os.path.join(save_path, model_name))

        running_loss += loss.item()

    # note len(dataloader) is number of batches
    epoch_loss = running_loss/len(dataloader) # len(dataloader) = no. of examples / batch size 
    time2 = datetime.datetime.now()

    return epoch_loss, (time2-time1).total_seconds()

def train_model(model, training_info, opt = None, start_epoch = 0): 
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
        train_loss, runtime = train_epoch(model, 
                                           training_info["train_loader"], 
                                           training_info["criterion"],
                                           training_info["optimizer"],
                                           epoch, 
                                           training_info["save_path"],
                                           mode = 'train')

        print("Epoch:%d, train loss: %.4f, time: %.2fs" %(epoch+1, train_loss, runtime))
        
        if (epoch + 1) % EVAL_EVERY_EPOCH == 0: 
            valid_loss, runtime = train_epoch(model, 
                                             training_info["valid_loader"], 
                                             training_info["criterion"],
                                             training_info["optimizer"],
                                             epoch, 
                                             training_info["save_path"],
                                             mode = 'validate')
                
            print('-'*60)
            print("Epoch:%d, valid loss: %.4f, time: %.2fs" %(epoch+1, valid_loss, runtime))
            print('-'*60)

            """
            CHECK EARLY STOPPING CONDITIONS
            """
            if valid_loss < MIN_LOSS:
                MIN_LOSS = valid_loss
                EARLY_STOPPING_COUNT = 0

                # save the best model so far 
                state = {"epoch": epoch, "model": model.state_dict(), "valid_loss": valid_loss, 
                        "train_loss": train_loss}
                model_name = "r2p_model_epoch{}.pth.tar".format(epoch)
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
                        help = "path to Flickr30k bounding box and phrase data.")
    parser.add_argument("--image_path", 
                        default = "/content",
                        help = "path to Flickr30k images data set.")
    parser.add_argument("--word_emb_path", 
                        default = "/content",
                        help = "path to Glove embeddings.")
    parser.add_argument("--emb_dim", default = 1024, type = float,
                        help = "Joint embedding dimension.")
    parser.add_argument("--activate_fn", default = "Tanh", type = str,
                        help = "Activation function for generating embeddings. Either Tanh, Leaky ReLU, Sigmoid or ReLU.")
    parser.add_argument("--batch_size", default = 64, type = int, 
                        help = "size of training mini-batch.")
    parser.add_argument("--learning_rate", default = 0.00005, type = float,
                        help = "initial learning rate.")
    parser.add_argument("--lr_decay", default = 0.95, type = float,
                        help = "gamma for learning rate scheduler.")
    parser.add_argument("--weight_decay", default = 1e-5, type = float,
                        help = "weight decay for optimizer.")
    parser.add_argument("--num_epochs", default = 50, type = int,
                        help = "number of training epochs.")
    parser.add_argument("--early_stop", default = 10, type = int, 
                        help = "number of epochs used for early stopping.")
    parser.add_argument("--save_path", 
                        default = "/content", 
                        help = "path to where the model weights are saved.")
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    opt = parser.parse_args()
    print(opt)

    ### load data from path 
    train_data = pd.read_csv(opt.data_path + "/flickr30k_entities_train_clean.csv", encoding= 'unicode_escape')
    valid_data = pd.read_csv(opt.data_path + "/flickr30k_entities_valid_clean.csv", encoding= 'unicode_escape')

    ### drop any duplicate rows in dataset 
    train_data = train_data.drop_duplicates()
    valid_data = valid_data.drop_duplicates()

    ### initialise word embedding model 
    print("Loading GLoVe Embeddings...")
    word_emb_model = init_glove_model(opt.word_emb_path) 

    ### create data loaders for training and validation set 
    train_dl = Flickr30kEntsLoader(train_data, opt.image_path, 
                               word_emb_model = word_emb_model)
    valid_dl = Flickr30kEntsLoader(valid_data, opt.image_path, 
                                word_emb_model = word_emb_model)

    train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = True, 
                                                batch_size = opt.batch_size, 
                                                collate_fn = collate_fn_static)

    valid_dataloader = torch.utils.data.DataLoader(valid_dl, shuffle = False, 
                                                    batch_size = opt.batch_size,
                                                    collate_fn = collate_fn_static)
    

    ### initialise model
    model = Region2PhraseModel(opt.emb_dim, image_model_type = "vision transformer",
                            word_emb_model = None,
                            word_emb_type = "static",
                            activate_fn = opt.activate_fn)

    ### move device to GPU
    model = model.to(device)

    ### optimizer and loss functions
    criterion = SymmetricLoss()
    optimizer = optim.Adam(model.parameters(), lr = opt.learning_rate, 
                            weight_decay = opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1.0, gamma = opt.lr_decay)

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
            train_model(model, training_info, start_epoch = start_epoch)
    else: 
        ### train model from scratch 
        print("Training model from scratch!")
        train_model(model, training_info)

if __name__ == '__main__':
    main()