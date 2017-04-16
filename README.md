# Tensorflow Implementation of Tree-based Convolutional Neural Network

Implementation of [TBCNN](https://sites.google.com/site/treebasedcnn/) using Tensorflow.

Instead of training on C programs, This version is used for SQL injection detection. Dataset is provided in the data folder.


### Installation
It's highly recommended to install into a virtual environment. Run the following command after activating venv:

`pip install -r requirements/dev.txt`

### Usage
`tbcnn.embedding` is the entry point for pretraining the embedding matrix.

`tbcnn.tbcnn` is the entry point for the tbcnn model. It requires a pre-trained embedding matrix.
