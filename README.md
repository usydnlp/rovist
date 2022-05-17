# RoViST: Learning Robust Metrics for Visual Storytelling

This repository contains code for paper [Learning Robust Metrics for Visual Storytelling](https://arxiv.org/pdf/2205.03774.pdf).

### <div align="center"> Wang, E.\*, Han, C.\*, & Poon, J. (2022). <br> [Learning Robust Metrics for Visual Storytelling](https://arxiv.org/pdf/2205.03774.pdf) <br> Findings of NAACL 2022 </div>

## 1. Introduction
Visual storytelling (VST) is the task of generating a story paragraph that describes a given image sequence. Most existing
storytelling approaches have evaluated their models using traditional natural language generation metrics like BLEU
or CIDEr. However, such metrics based on n-gram matching tend to have poor correlation with human evaluation scores
and do not explicitly consider other criteria necessary for storytelling such as sentence structure or topic coherence. Moreover, a single score is not enough to assess a story as it does not inform us about what specific errors were made by the
model. 

In this work, we propose 3 evaluation metrics sets that analyses which aspects we would look for in a good story: 
* **Visual Grounding:** generating text relevant to the image content but unlike image captioning, there is less emphasis on describing relationships between objects and may contain concepts that are inferred from the image.
* **Coherence:** the story must be topically coherent, similar to how a human would tell a story in a social setting. Sentences should not sound disjointed e.g. ‘We went to the park. I grew up in Sydney’.
* **Non-redundancy:** avoids repetition which appears to be a common issue in current VST models e.g. ‘we had a good time and had a great time!’

## 2. Setup
As the code format is .ipynb, there are no settings but the Jupyter notebook with GPU.

## 3. Inference Notebook
To calculate your own scores, follow the instructions in the Demo notebook files. The code can be run on Google Colab. **RoViST_VG_Demo** can be used to calculate the Visual Grounding scores and **RoViST_C_NR** can be used to calculate the Coherence and Non-redundancy scores. 

## 4. Reference
If you use this code for your research, please cite:
```
@article{wang2022rovist,
  title={RoViST: Learning Robust Metrics for Visual Storytelling},
  author={Wang, Eileen and Han, Caren and Poon, Josiah},
  journal={arXiv preprint arXiv:2205.03774},
  year={2022}
}
```
