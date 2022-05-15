import json
import pandas as pd
import os 
import nltk
import argparse 
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


def extract_nouns(output_stories, output_nouns_fname):
    """
    Extract nouns from the output story and saves to json file. 
    """
    data_dict = {} 

    # tags_to_consider = set(["NN", "NNS"])
    tags_to_consider = set(["NN", "NNS", "NNP", "NNPS"])
    for story_id in output_stories: 
        
        nouns_in_story = []

        story_sents = nltk.sent_tokenize(output_stories[story_id])

        for sent in story_sents: 

            nouns_in_image = set() # get unique nouns per sent per image

            # odd cases 
            sent = sent.replace("[male]", "male")
            sent = sent.replace("[female]", "female")
            sent = sent.replace("[location]", "location")
            sent = sent.replace("[organization]", "organization")

            pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
        
            for token, tag in pos_tags: 
                
                if token == 'i': continue # ignore personal pronoun I 

                if tag in tags_to_consider: 
                    nouns_in_image.add(token)
            
            nouns_in_story += list(nouns_in_image)

        data_dict[story_id] = nouns_in_story
        
        with open(output_nouns_fname, 'w') as fp:
            json.dump(data_dict, fp)

def main(): 

    parser = argparse.ArgumentParser() 

    parser.add_argument("--output_story_file", 
                        default = "/content/arel_stories.json",
                        help = "Path to json file containing output machine stories.")
    parser.add_argument("--output_noun_file_name", 
                        default = "arel_nouns.json",
                        help = "Name of output json file containing the extracted nouns for each story id.")

    opt = parser.parse_args()

    # load machine stories 
    output_stories = json.load(open(opt.output_story_file))

    extract_nouns(output_stories, opt.output_noun_file_name)

if __name__ == '__main__':
    main()