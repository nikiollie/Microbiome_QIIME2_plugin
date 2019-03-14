import numpy as np
import os
import csv
import sys
import skbio
import pickle
import random
labels = []

def read_fasta(fp):
        name, seq = None, []
        for line in fp:
            #remove first line from file
            line = line.rstrip()
            if line.startswith(">"):
                #create a string
                if name: yield (name, ''.join(seq))
                name, seq = line, []
            else:
                seq.append(line)
        if name: yield (name, ''.join(seq))

def vectorizeSequence(seq):
    # the order of the letters is not arbitrary.
    # Flip the matrix up-down and left-right for reverse compliment
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],
            'a':[1,0,0,0],'c':[0,1,0,0],'g':[0,0,1,0],'t':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])
    #return [[ltrdict[x] for x in seq]]
#directory = os.getcwd()
#directory = os.fsencode(directory1)
directory = os.path.dirname(os.path.realpath(__file__))
data_directory = directory + "/fna_files/"
pickle_directory = directory + "/pickle_files/"
data_directory_encode = os.fsencode(data_directory)
counter = 0
dataset = []
for file in os.listdir(data_directory_encode):
    filename = os.fsdecode(file)
    if filename.endswith(".fna"):
        label = list(skbio.io.read(data_directory + filename,
            format='fasta'))[0].metadata['id']
        labels.append(label)
        data_filename = data_directory + filename
        with open(data_filename) as fp:
            for name, seq in read_fasta(fp):
                i = 0
            one_hot_seq = vectorizeSequence(seq)
        #length of the DNA sample
        seqLength = len(one_hot_seq)
        #list to store the 10000 150x4 training examples
        training_examples = []
        num_labels = [[counter]]*10000
        num_labels = np.asarray(num_labels)
        
        counter+=1
        labels_ex = []
        for i in range(0,10000):
           
            #generates 150 random integers to parse the DNA sample
            indices  = np.random.randint(seqLength,size = 150)
    	    #a single 150x4 random example
            example = one_hot_seq[indices,:]
            training_examples.append(example)
        #create an array with 10K elements for 10K rows
        training_examples = np.asarray(training_examples)
        outer_arr=[]
        for j in range(10000):
            outer_arr.append([training_examples])
            outer_arr[j]+=[[num_labels]]
        random_indices = np.random.randint(10000,size = 1000)
        for m in range(1000):
            dataset.append(outer_arr[random_indices[m]])
    else:
        continue
    file1 = open(pickle_directory + str(label) + '.pickle', 'wb')
    pickle.dump(training_examples,file1)
    file1.close()
print(np.shape(dataset[0][0]))
onehot_labels = np.matlib.identity(len(labels))
pickle_labels = open(pickle_directory + 'labels' + '.pickle', 'wb')
pickle.dump(onehot_labels, pickle_labels)
pickle_labels.close()
