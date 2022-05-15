import json
import os
import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F

import nltk
import argparse
nltk.download('punkt')

from rovist_c import SOPClassifier, model_nm, tokenizer

def coherence_score(selected_stories, cand_stories, model, device): 

    coherence_scores = [] 

    for key in selected_stories:
        
        print("Evaluating {}".format(key))

        text = cand_stories[key]
        sentences = nltk.sent_tokenize(text)

        scores = []

        for i in range(len(sentences)-1): 
            sentA = sentences[i]
            sentB = sentences[i+1]

             # repeated sentence --> automatically assign 0 score
            if sentA.strip() == sentB.strip():
              scores.append(0)
              continue 
            
            sentA_len = len(tokenizer.tokenize(sentA))
            sentB_len = len(tokenizer.tokenize(sentB))

            text_list = ['[CLS]', sentA, '[SEP]', sentB, '[SEP]']
            input_text = ' '.join(text_list)

            tokenized_text = tokenizer.tokenize(input_text)

            # token ids
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            token_ids = torch.tensor([indexed_tokens])
            
            # sequence ids
            segment_ids = (sentA_len+2)*[0] + (sentB_len+1) * [1]
            segment_ids = torch.tensor([segment_ids])

            # attention mask id 
            attention_ids = (sentA_len+2)*[1] + (sentB_len+1) * [1]
            attention_ids = torch.tensor([attention_ids])

            # if GPU is available, move data to GPU 
            token_ids = token_ids.to(device)
            attention_ids = attention_ids.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                sop_logits = model(token_ids, attention_ids, segment_ids)

            sop_probs = F.softmax(sop_logits, dim=1)

            # probability of being coherent is stored at index 1
            scores.append(sop_probs[0][1].item())

        coherence_scores.append(np.mean(scores))
      
    return coherence_scores

def main(): 

    parser = argparse.ArgumentParser() 

    parser.add_argument("--model_checkpoint", 
                        default = "/content/sop_model_epoch4.pth.tar",
                        help = "path to pre-trained ALBERT model.")
    parser.add_argument('--selected_stories', 
                        default = None, 
                        help='story IDs to evaluate; delimited list input', type=str)
    parser.add_argument('--output_stories_path', 
                        default = "/content/arel_stories.json", 
                        help='json file containing the machine-generated stories.', type=str)

    opt = parser.parse_args()

    ### Enable GPU
    use_cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('We are using GPU.' if use_cuda else 'We are using CPU.')

    ### load output stories from json file  
    output_stories = json.load(open(opt.output_stories_path))

    if opt.selected_stories: 
        selected_stories = [x for x in opt.selected_stories.split(',')]
    else: 
        selected_stories = list(output_stories.keys()) # evaluate all stories 

    ### load model
    if os.path.isfile(opt.model_checkpoint): 
        print("Loading pre-trained ALBERT model.")
        checkpoint = torch.load(opt.model_checkpoint)
        ckpt_opt = checkpoint["opt"]

        model = SOPClassifier(ckpt_opt.hidden_dim, ckpt_opt.dropout_prob,
                                model_nm)
        model = model.to(device) # move model to GPU

        model.load_state_dict(checkpoint["model"])
        model.eval()
    else: 
        print("No checkpoint find at this path!")
        return 

    ### compute coherence scores
    final_scores = coherence_score(selected_stories, output_stories, model, device)

    ### save final scores to dataframe 
    df = pd.DataFrame()
    df["story_id"] = selected_stories
    df["coherence_score"] = final_scores
    df.to_csv("coherence_scores.csv", index = False)

    ### system level score 
    print("Coherence Score: {}".format(df["coherence_score"].mean()))

if __name__ == '__main__':
    main()