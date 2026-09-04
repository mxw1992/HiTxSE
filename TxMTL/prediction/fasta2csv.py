# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 21:21:01 2025

@author: Lenovo
"""
import pandas as pd
import os


def fasta2csv(fasta_path):
    with open(fasta_path, 'r') as f:
        content = f.read()
        seq = content.split('>')
    del(seq[0])


    df = pd.DataFrame(columns=['ID', 'SEQUENCE', 'SEQUENCE_space'])

    seq_name =[]
    for i in range(len(seq)):
        a = seq[i].split('\n')
        df.loc[i] = [a[0], a[1].strip(), " ".join(a[1].strip())]
        seq_name.append(a[0])
    df.to_csv('./test_seq.csv',index=False)
    return seq_name, os.getcwd()+ '/test_seq.csv'
