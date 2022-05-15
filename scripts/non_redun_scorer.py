import nltk
import argparse
import json
import numpy as np
import pandas as pd

nltk.download("punkt")

def jaccard_sim(selected_stories, cand_stories, ngram_len): 
    """
    Calculate the repetition score for all output stories. 
    """
    story_rep_scores = []

    for key in selected_stories: 

        story = cand_stories[key]
        sent_tokens = nltk.sent_tokenize(story)

        word_tokens = [] 

        for sent in sent_tokens: 
            # don't include the punctuation at the end. 
            word_tokens.append(nltk.word_tokenize(sent)[:-1])

        # check inter sentence repetition. 
        inter_sentence_scores = [] 

        for i in range(1, len(word_tokens)): 
            next_ngrams = word_tokens[i]

            for j in range(0, i): 
                prev_ngrams = word_tokens[j]
                union = len(set(next_ngrams + prev_ngrams))

                intersection = 0
                for token_i in next_ngrams: 
                    for token_j in prev_ngrams: 
                        if token_i == token_j: 
                            intersection += 1

                inter_sentence_scores.append(intersection/union)

        # check intra sentence repetition
        intra_sentence_scores = [] 
        
        for tokens in word_tokens: 
            for i in range(0, len(tokens), ngram_len): 
                j = i + ngram_len
                prev_slice = tokens[i:i+ngram_len] 
                next_slice = tokens[j:j+ngram_len]
        
                if len(next_slice) == 0: continue 
                union = len(set(prev_slice + next_slice))
                
                intersection = 0
                for token_i in prev_slice: 
                    for token_j in next_slice: 
                        if token_i == token_j:
                            intersection += 1
                
                intra_sentence_scores.append(intersection/union)

        if len(intra_sentence_scores) != 0: 
            repetition_score = [np.mean(inter_sentence_scores), np.mean(intra_sentence_scores)]
        else: # sentences are way too short to do intra sentence rep 
            repetition_score = [np.mean(inter_sentence_scores), 0]

        story_rep_scores.append(1-np.mean(repetition_score))

    return story_rep_scores

def main():

    parser = argparse.ArgumentParser() 

    parser.add_argument('--selected_stories', 
                        default = None, 
                        help='story IDs to evaluate; delimited list input', type=str)
    parser.add_argument('--output_stories_path', 
                        default = "/content/arel_stories.json", 
                        help='path to json file containing machine-generated stories.', type=str)
    parser.add_argument('--intra_sent_ngram_len', 
                        default = 4,   
                        help='n-gram length for intra-sentence coherence score.', type=int)

    opt = parser.parse_args()

    ### load output stories from json file  
    output_stories = json.load(open(opt.output_stories_path))

    if opt.selected_stories: 
        selected_stories = [x for x in opt.selected_stories.split(',')]
    else: 
        selected_stories = list(output_stories.keys()) # evaluate all stories 

    final_scores = jaccard_sim(selected_stories, output_stories, opt.intra_sent_ngram_len)

    ### save final scores to dataframe 
    df = pd.DataFrame()
    df["story_id"] = selected_stories
    df["non_redun_score"] = final_scores
    df.to_csv("non_redun_scores.csv", index = False)

    ### system level score 
    print("Non Redundancy Score: {}".format(df["non_redun_score"].mean()))

if __name__ == '__main__':
    main()