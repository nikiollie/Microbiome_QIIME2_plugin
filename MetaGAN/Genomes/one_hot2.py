import numpy as np

def read_fasta(fp):
        name, seq = None, []
        for line in fp:
            line = line.rstrip()
            if line.startswith(">"):
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
fileInput = input("enter file: ")
with open(str(fileInput)) as fp:
    for name, seq in read_fasta(fp):
        i = 0
    #one hot encoded DNA sequence
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
labels = np.asarray([str(labels)]*10000)
training_examples = np.asarray(training_examples)
#print(labels.shape)
#print(labels)
#print(training_examples.shape)
#print(training_examples)
outer_arr=[]

for i in range(1):
    for j in range(10000):
        outer_arr.append([[labels[j]]])
for k in range(len(outer_arr)):
    #print(k)
    outer_arr[k]+=[training_examples[k]]
#outer_arr=np.asarray(outer_arr)
print(outer_arr)

#labeled_examples = np.vstack((training_examples,labels))
#training_examples = np.asarray(labeled_examples_list)
