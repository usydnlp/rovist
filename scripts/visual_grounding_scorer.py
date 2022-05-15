from rovist_vg import Region2PhraseModel, init_glove_model
import pickle
import argparse 
import pandas as pd 
import json
import os
import torch
import torch.nn.functional as F
from PIL import Image

use_cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('We are using GPU.' if use_cuda else 'We are using CPU.')

def nouns_and_imageids(story_id, output_nouns, all_stories, idf_dict = None): 
    """
    Return list of image IDs, nouns, and idf weights of nouns for given story id. 
    
    Args: 
        story_id: given as string.
        output_nouns: dictionary of story IDs and nouns 
        all_stories: dictionary of ground truth stories with image sequence
        idf_dict: If idf scores provided, extract and use as weights. Else all weights are 1. 
    """

    nouns = output_nouns[story_id]
    image_ids = all_stories[story_id]['images'] 
    image_ids = [int(x) for x in image_ids]

    
    if idf_dict is not None: 

        weights = []

        for n in nouns: 
            weights.append(idf_dict[n])
        weights = torch.tensor(weights)
    
    else: 

        weights = [1] * len(nouns)
        weights = torch.tensor(weights)
    
    assert len(weights) == len(nouns)

    return nouns, image_ids, weights

def get_text_embs(nouns, model, word_emb_model, norm = 2): 
    """
    Feed nouns through text embedder. 
    """
    embs_list = []

    for word in nouns: 
        try: 
            embs_list.append(torch.tensor(word_emb_model[word]))
        except: 
            embs_list.append(torch.tensor(word_emb_model["unk"]))
                    
    embs = torch.stack(embs_list)

    # feed noun phrases through text encoder
    batch_data = {"phrase_embeddings": embs}

    with torch.no_grad():
        text_embs = model.text_encoder(batch_data) # (num noun phrases, 1024)
  
    # normalise embeddings 
    if norm is not None: 
        text_embs = F.normalize(text_embs, p = norm, dim = -1)

    return text_embs 


def compute_sim(text_embs, image_ids, model, vist_entities_df, img_path, norm = 2,
                num_bbox = 10):
    """
    Returns cosine similarity recall. 

    Args:
        text_embs: text embeddings of nouns.
        image_ids: list of image ids used in the photo sequence for the story.
        model: rovist_vg model.
        vist_entities_df: dataframe containing vist image id with bbox info.
        num_bbox: number of bounding boxes per image to consider.
    """

    num_images = len(image_ids)

    final_sims = torch.zeros((num_images, text_embs.shape[0]))
    txt2reg_inds = torch.zeros((num_images), text_embs.shape[0])

    all_imgs = [] # nested list containing PIL images = (nnum_bbox times 5)

    for i in range(len(image_ids)):


        img_id = image_ids[i] 
      
        df = vist_entities_df[vist_entities_df["image_id"] == img_id]
        bboxes = df["bbox"].tolist()

        img_list = [] # stores list of PIL images 
        for bbox in bboxes: 

            img_name = str(img_id) + "_" + bbox + ".jpg"
            img = Image.open(img_path + '/' + img_name).convert('RGB')
            img_list.append(img)

        all_imgs.append(img_list)

        with torch.no_grad(): # turn off gradient calculation
            # feed list of PIL images through image encoder 
            img_embs = model.image_encoder(img_list)[:num_bbox] # (num_bbox, 1024)

        if norm is not None: 
            img_embs = F.normalize(img_embs, p = norm, dim = -1) # (nnum_bbox, 1024)

        sim_matrix = text_embs @ img_embs.T # (number of nouns, num_bbox)

        img_recall = torch.max(sim_matrix, dim = 1)[0].cpu() # (number of nouns)
        img_recall_inds = torch.max(sim_matrix, dim = 1)[1].cpu()

        final_sims[i] = img_recall 
        txt2reg_inds[i] = img_recall_inds

    recall = torch.max(final_sims, dim = 0)[0]
    recall_inds = torch.max(final_sims, dim = 0)[1]
    
    return recall, recall_inds, txt2reg_inds, all_imgs


def vg_score(x, idf_weights, aggregate = "log_sum_exp"): 
    """
    Compute final recall score. 
    """

    x = x * idf_weights

    if aggregate  == "log_sum_exp":
  
        score = torch.log(torch.exp(x).sum())

    elif aggregate == "sum":

        score = x.sum()
        
    elif aggregate == "mean": 
        score = x.mean() 

    return score 

def main():

    parser = argparse.ArgumentParser() 

    parser.add_argument("--path_to_nouns", 
                        default = "/content/MCSM_nouns.json",
                        help = "path to output stories with extracted nouns.")
    parser.add_argument("--path_to_vist_bbox_info", 
                        default = "/content",
                        help = "path to csv file containing image ids matched with bbox information.")
    parser.add_argument('--selected_stories', 
                        default = None, 
                        help='story IDs to evaluate; delimited list input', type=str)
    parser.add_argument('--path_to_story_info', 
                        default = "/content", 
                        help='path to json files containing ground truth stories.')
    parser.add_argument('--path_to_idf_dict', 
                        default = "/content", 
                        help='path to pickle file containing idf scores.')
    parser.add_argument('--use_idf', 
                        default = 1, 
                        help='1 is weight by idf score, else 0.')
    parser.add_argument('--img_path', 
                        default = "/content/vist-entities", 
                        help='path to vist bounding box images.')
    parser.add_argument('--path_to_glove', 
                        default = "/content", 
                        help='path to glove embeddings.')
    parser.add_argument('--model_checkpoint', 
                        default = "/content/r2p_model_epoch2.pth.tar", 
                        help='rovist_vg model checkpoint.')

    opt = parser.parse_args()

    # load extracted nouns from machine story 
    output_nouns = json.load(open(opt.path_to_nouns))

    # vist image ids and bbox info
    vist_entities_df = pd.read_csv(opt.path_to_vist_bbox_info + "/vist_entities.csv")

    ### load ground truth story info
    test_stories = json.load(open(os.path.join(opt.path_to_story_info, "test_stories.json")))
    valid_stories = json.load(open(os.path.join(opt.path_to_story_info, "valid_stories.json")))
    train_stories = json.load(open(os.path.join(opt.path_to_story_info, "train_stories.json")))

    all_stories = {**test_stories, **valid_stories, **train_stories}

    ### load glove embedding model. Takes a couple of minutes. 
    print("Loading GloVe embeddings.")
    word_emb_model = init_glove_model(opt.path_to_glove) 

    ### initialise model
    model = Region2PhraseModel(1024, image_model_type = "vision transformer",
                                word_emb_model = None,
                                word_emb_type = "static")

    model = model.to(device)

    ### load from check point 
    if os.path.isfile(opt.model_checkpoint):
        print("Loading pre-trained model!!!")
        checkpoint = torch.load(opt.model_checkpoint)
        model.load_state_dict(checkpoint['model'])
        model.eval() 

    if opt.selected_stories: 
        selected_stories = [x for x in opt.selected_stories.split(',')]
    else: 
        selected_stories = list(output_nouns.keys()) # evaluate all stories 

    #print(selected_stories)

    if bool(opt.use_idf) == True: 
        ### load idf dict 
        print("Using IDF weighting to compute score.")
        idf_dict = pickle.load(open(opt.path_to_idf_dict + '/idf_dict.pkl', 'rb'))
    else: 
        idf_dict = None 

    ### evaluate stories 
    final_scores = []

    for story_id in selected_stories: 

        print("Evaluating {}".format(story_id))

        nouns, image_ids, weights = nouns_and_imageids(story_id, output_nouns, all_stories, idf_dict)
        text_embs = get_text_embs(nouns, model, word_emb_model)
        recall, recall_inds, txt2reg_inds, images = compute_sim(text_embs, 
                                                                image_ids, 
                                                                model, 
                                                                vist_entities_df,
                                                                opt.img_path,
                                                                num_bbox = 10)
        score = vg_score(recall, weights, aggregate = "log_sum_exp").item()  
        final_scores.append(score)

    ### save final scores to dataframe 
    df = pd.DataFrame()
    df["story_id"] = selected_stories
    df["vg_score"] = final_scores
    df.to_csv("vg_scores.csv", index = False)

    ### system level score 
    print("Visual Grounding Score: {}".format(df["vg_score"].mean()))

if __name__ == '__main__':
    main()