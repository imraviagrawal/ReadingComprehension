# Reading Comprehensiom

Context paragraph: On 24 March 1879, Tesla was returned to Gospic under police guard for
not having a residence permit. On 17 April 1879, Milutin Tesla died at the age of 60 after
contracting an unspecified illness (although some sources say that he died of a stroke). During
that year, Tesla taught a large class of students in his old school, Higher Real Gymnasium, in
Gospic.

Question: Why was Tesla returned to Gospic?

Answer : not having a residence permit


## The project has several dependencies that have to be satisfied before running the code. You can install them using your preferred method -- we list here the names of the packages using `pip`.

# Requirements

The starter code provided pressuposes a working installation of Python 2.7, as well as a TensorFlow 0.12.1.

It should also install all needed dependnecies through
`pip install -r requirements.txt`.

# Running 

The WordEmbedding and the preprocessing scripts are taken from the Stanford CS224 class, though we will be modelling and training different neural network models.

You can get started by downloading the datasets and doing dome basic preprocessing:

$ code/get_started.sh

Note that you will always want to run your code from this assignment directory, not the code directory, like so:

$ python code/train.py

This ensures that any files created in the process don't pollute the code directoy.

# Dataset
After the download, the SQuAD dataset is placed in the data/squad folder. SQuAD downloaded
files include train and dev files in JSON format:
• train-v1.1.json: a train dataset with around 87k triplets.
• dev-v1.1.json: a dev dataset with around 10k triplets.

Note that there is no test dataset publicly available: it is kept by the authors of SQuAD to ensure fairness in model evaluations. While developing the model, we will consider for all purposes the dev set as our test set, i.e., we won’t be using the dev set until afterinitial model development. Instead, we split the supplied train dataset into two parts: a 95% slice for training, and the rest 5% for validation purposes, including hyperparameter search. We refer to these as train.* and val.* in filenames.

