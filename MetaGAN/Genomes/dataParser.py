import sys
import csv
import os
import skbio

def dataParser(input_file):
    #remove '.fna' from file name string
    name_file = os.path.splitext(input_file)[0]
    
    sequences = list(skbio.io.read(input_file, format='fasta')) 
    
    sequence = ""
    #open file and write to it(modify file)
    #with open('MetaGAN/Edited_Genomes/Edited_' + name_file + ".fna", 'w') as fout:
    for value in sequences[0].values:
        sequence += str(value.decode('UTF-8')) 
    
    return sequence

#state the directory where the genomes are located
#directory_in_str = 'MetaGAN/Genomes'

#encode path-like filename to the filesystem encoding
#return bytes unchanged
#directory = os.fsencode(directory_in_str)

#listdir function returns a list containing the names 
#of the entries in the directory given by the path
#iterate through every file in the directory
#for file in os.listdir(directory):
    
    #fsdecode function decodes the path-like filename 
    #from the filesystem encoding and returns the str unchanged 
    #filename = os.fsdecode(file)
    #if filename.endswith(".fna"): 
        # print(os.path.join(directory, filename))
        #call dataParser function 
        #dataParser(filename)
        #continue
    #else:
        #continue



