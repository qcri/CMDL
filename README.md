# CMDL
Cross-Modal Data Discovery over Structured and Unstructured Data Lakes

Set up:
- environment.yml will set up a conda environment

Entry points:
- trainer/pretrain-text.ipynb: Fine tuning a language model on text corpus to learn text embeddings

- trainer/pretrain-tables.ipynb: Fine tuning a language model on table collection to learn tuple embeddings

- trainer/column_text_joint_training.ipynb: training a baseline connecting text to table columns

- compare_gt.py: accuracy measurement of search based baselines and similarity sketches on text->table relation discovery using the ground truth provided

Data Sets & Ground Truths:
- The `Pharma` dataset referred to in the paper is in the `inputs` directory with the tables in drugbank-tables and text documents in the pubmed-targets subdirectories respectively.
- The (`MLOpen`)[] & (`UKOpen`)[] datasets referred in the paper can be found here. 
- The ground truth files for each dataset are present in the `inputs` directory

Resources:
- Paper manuscripts provided under the folder 'docs'

Prior baselines:
- snorkel labeler.ipynb needs to be run in its separate environment by following instructions at: https://github.com/snorkel-team/snorkel

- build_label_files.py: profiles data, indexes tables, creates labels by probing indexes using each text

- build_features.py: featurizes input data, saves features to disk to be read during training



