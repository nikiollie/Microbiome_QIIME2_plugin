import sys
import csv
import requests
import os
from Bio import SeqIO

def dataParser(input_file):
    base = input_file
    name_file = os.path.splitext(base)[0]

    f = open(name_file + ".txt", "w")

    with open(input_file, "r") as fin:
        data = fin.read().splitlines(True)
    with open(f, 'w') as fout:
        fout.writelines(data[1:]) 


directory_in_str = '/MetaGAN/Genomes'
directory = os.fsencode(directory_in_str)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".fna"): 
        # print(os.path.join(directory, filename))
        dataParser(filename)
        continue
    else:
        continue



