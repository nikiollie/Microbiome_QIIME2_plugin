import sys
import csv
import os


def dataParser(input_file):
    base = input_file
    #remove '.fna' from file name string
    name_file = os.path.splitext(base)[0]
    
    #create a new file and write to it(modify file)
    f = open('MetaGAN/Edited_Genomes/' + name_file + ".txt", "w")

    #open file and read to it(display)
    with open('MetaGAN/Genomes/' + name_file + ".fna", "r") as fin:
        data = fin.read().splitlines(True)
    #open file and write to it(modify file)
    with open('MetaGAN/Edited_Genomes/' + name_file + ".txt", 'w') as fout:
        #remove first line from file
        fout.writelines(data[1:]) 

#state the directory where the genomes are located
directory_in_str = 'MetaGAN\Genomes'

#encode path-like filename to the filesystem encoding
#return bytes unchanged
directory = os.fsencode(directory_in_str)

#return a list containing the names of the entries in the directory 
#given by path
for file in os.listdir(directory):
    
    #Decode the path-like filename from the filesystem encoding
    #return str unchanged 
    filename = os.fsdecode(file)
    if filename.endswith(".fna"): 
        # print(os.path.join(directory, filename))
        #call dataParser function 
        dataParser(filename)
        continue
    else:
        continue



