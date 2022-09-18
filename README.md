# Prepare Environment

- build docker image with ```docker build . -t sem2vec```
- all the following commands can be executed in the container.

# Preprocess Data

We have already prepared the preprocessed data in the codebase (see ```data/constraints.txt```, ```data/pair```, ```FoBERT/merges.txt``` and ```FoBERT/vocab.json```)

To use your own data, please use the following steps.
- pretraining data: use in-order traversal of constraints and run ```python data/preprocess.py raw_constraints.txt constraints.txt```.
- fine-tuning data: follow the above commands to preprocess constraints and form constraint pairs with corresponding labels (whether from the same line).

# Train Model

We pretrain and fine-tune the model on NVIDIA 3090. It may encounter out-of-memory problems if the GPU memory is not large enough.

- pretraintrain RoBERTa model: ```python src/run_roberta.py```
- fine-tune RoBERTa model: ```python src/fine_tune.py```

# Mask Prediction and Embedding Generation

We show how to use the pretrained model to predict the masked token in line 50-57 of ```src/run_roberta.py``` and use the fine-tuned model to generate the embedding of constraints in line 54-58 of ```src/fine_tune.py```