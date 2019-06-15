# README

To install the Tensorflow virtual environment, go to https://www.tensorflow.org/install/pip and follow the instructions based on your OS. The Tensorflow virtual environment contains essential Tensorflow libraries that allow us to run our model. 

To install the QIIME2 virtual environment, follow the instructions at https://docs.qiime2.org/2019.4/install/native/#install-qiime-2-within-a-conda-environment. The QIIME2 environment is needed to process our data and run our one_hot.py file.

In order to run or modify our code, you must have data files (.fna) which contain DNA sequences of certain classes. Currently these are in the MetaGAN/Genomes/fna_files directory. You need to activate the QIIME2 virtual environment and then you can run our data pre-processing file, one_hot.py (in MetaGAN/Genomes  directory) with "python one_hot.py" which will one hot encode the DNA sequences and 'pickle' them. Next you have to activate the Tensorflow virtual environment so that you can run our model (in MetaGAN/Genomes directory) with "python model.py" which will then output the training and validation accuracies for each epoch.

