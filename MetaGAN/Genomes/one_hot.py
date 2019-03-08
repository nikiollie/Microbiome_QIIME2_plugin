import numpy as np
import os
import csv
import sys
import skbio
import pickle
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
for file in os.listdir(data_directory_encode):
    filename = os.fsdecode(file)
    if filename.endswith(".fna"):
        data_filename = data_directory + filename
        with open(data_filename) as fp:
            for name, seq in read_fasta(fp):
                print(name)
                i = 0
            one_hot_seq = vectorizeSequence(seq)
        #length of the DNA sample
        seqLength = len(one_hot_seq)
        #list to store the 10000 150x4 training examples
        training_examples = []
        for i in range(0,10000):
            #generates 150 random integers to parse the DNA sample
            indices  = np.random.randint(seqLength,size = 150)
    	    #a single 150x4 random example
            example = one_hot_seq[indices,:]
            training_examples.append(example)
        #create an array with 10K elements for 10K rows
        training_examples = np.asarray(training_examples)
    else:
        continue
    file1 = open(pickle_directory + str(file) + '.pickle', 'wb')
    #file1 = open(str(file)+'.pickle', 'wb')
    pickle.dump(training_examples,file1)
    file1.close()
