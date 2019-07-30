# PathNet: Exploiting Explicit Paths for Multi-hop Reading Comprehension


This repository contains the source code for the paper [Exploiting Explicit Paths for Multi-hop Reading Comprehension](https://arxiv.org/abs/1811.01127).
This work was published at Association of Computational Linguistics (ACL) 2019.
If you find the paper or this repository helpful in your work, please use the following citation:

```
@inproceedings{pathnet,
  title={ Exploiting Explicit Paths for Multi-hop Reading Comprehension },
  author={ Souvik Kundu and Tushar Khot and Ashish Sabharwal and Peter Clark },
  booktitle={ ACL },
  year={ 2019 }
}
```

### Setup

We used Python-3.6.2. Consider creating a virtual/conda environment for development.
This code repository is built using [AllenNLP](https://github.com/allenai/allennlp).
To install all dependencies please run the following:
```bash
sh scripts/install_requirements.sh
```


### Download

To download all the required files, run `scripts/download.sh`


### Prediction for WikiHop

Once you run the `scripts/download.sh`, you should have our pretrained model for WikiHop
in `data/datasets/WikiHop/pretrained-model/`.
For generating the predictions using this model, follow the steps given in `scripts/predict_wikihop.sh`.


### Training

Follow the steps in `scripts/run_full_wikihop.sh` and `scripts/run_full_obqa.sh` for training new models
for WikiHop and OBQA, respectively.


### Path Extraction Demo

Run the `scripts/path_finder_wrapper.py` for simply visualizing the paths.

```
>>> from scripts.path_finder_wrapper import find_paths
>>> documents = ["...this is doc 1 ...", "...this is doc 2 ...", ...]
>>> question = "question text"
>>> candidate = "candidate text"
>>> pathlist = find_paths(documents, question, candidate, style="plain")
