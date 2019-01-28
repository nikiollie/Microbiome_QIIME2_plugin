import sys
import csv
import os


def dataParser(input_file):
    base = input_file
    name_file = os.path.splitext(base)[0]

    f = open('MetaGAN/Edited_Genomes/' + name_file + ".txt", "w")

    with open('MetaGAN/Genomes/' + name_file + ".fna", "r") as fin:
        data = fin.read().splitlines(True)
    with open('MetaGAN/Edited_Genomes/' + name_file + ".txt", 'w') as fout:
        fout.writelines(data[1:]) 


directory_in_str = 'MetaGAN\Genomes'
directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".fna"): 
        # print(os.path.join(directory, filename))
        dataParser(filename)
        continue
    else:
        continue



