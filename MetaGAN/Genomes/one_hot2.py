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
    ltrdict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    return np.array([ltrdict[x] for x in seq])
    #return [[ltrdict[x] for x in seq]]
#accepts input file from console (.fna)
#fileInput = input("enter file: ")
directory1 = os.getcwd()

directory = os.fsencode(directory1)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".fna"):
        with open(filename) as fp:
            #print(filename)
            for name, seq in read_fasta(fp):
                #print(filename)
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

        #accepts input label from console
        labels = input("enter label: ")
        #create an array with 10K elements for 10K rows
        labels = np.asarray([str(labels)]*10000)
        training_examples = np.asarray(training_examples)
    else:
        continue
    file1 = open(str(file)+'.pickle', 'wb')
    pickle.dump(training_examples,file1)
    file1.close()

