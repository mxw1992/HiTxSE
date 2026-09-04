#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

from transformers import T5EncoderModel, T5Tokenizer


# =========================================================
# clean sequence
# =========================================================

def clean_protein_sequence(seq):

    seq = str(seq).upper()
    seq = "".join(seq.split())

    seq = seq.replace("U", "X")
    seq = seq.replace("Z", "X")
    seq = seq.replace("O", "X")
    seq = seq.replace("B", "X")

    return seq



# =========================================================
# load ProtT5
# =========================================================

def load_prott5(
        model_name_or_path="Rostlab/prot_t5_xl_uniref50",
        device=None):


    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )


    tokenizer = T5Tokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=False
    )


    model = T5EncoderModel.from_pretrained(
        model_name_or_path
    )


    model = model.to(device)

    model.eval()


    return model, tokenizer, device



# =========================================================
# FASTA reader
# =========================================================

def read_fasta(fasta):

    names = []
    sequences = []


    for record in SeqIO.parse(fasta, "fasta"):

        names.append(record.id)

        sequences.append(
            clean_protein_sequence(record.seq)
        )


    return names, sequences



# =========================================================
# mean pooling
# =========================================================

def mean_pooling(
        token_embedding,
        seq_len):


    # 去掉special token
    residue_embedding = token_embedding[:seq_len]


    embedding = residue_embedding.mean(
        dim=0
    )


    return embedding



# =========================================================
# extract embedding
# =========================================================

def extract_embedding(
        fasta,
        output,
        seq_length=2000,
        batch_size=1,
        model_name="Rostlab/prot_t5_xl_uniref50"
):


    names, sequences = read_fasta(fasta)


    print(
        f"Loaded sequences: {len(sequences)}"
    )


    model, tokenizer, device = load_prott5(
        model_name
    )


    embeddings = []



    for start in tqdm(
        range(0,len(sequences),batch_size),
        desc="Extracting ProtT5"
    ):


        batch_seq = sequences[
            start:start+batch_size
        ]


        batch_seq = [
            x[:seq_length]
            for x in batch_seq
        ]


        # ProtT5格式:
        # A C D E F ...

        batch_seq = [
            " ".join(list(x))
            for x in batch_seq
        ]


        tokens = tokenizer.batch_encode_plus(
            batch_seq,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=seq_length+1,
            return_tensors="pt"
        )


        input_ids = tokens["input_ids"].to(device)

        attention_mask = tokens[
            "attention_mask"
        ].to(device)



        with torch.no_grad():

            output_hidden = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )


        hidden = output_hidden.last_hidden_state


        for i,seq in enumerate(batch_seq):

            real_len = len(
                seq.split()
            )


            emb = mean_pooling(
                hidden[i],
                real_len
            )


            embeddings.append(
                emb.cpu()
            )



    embeddings = torch.stack(
        embeddings
    )


    print(
        "Embedding shape:",
        embeddings.shape
    )


    save = {

        "names":names,

        "sequences":sequences,

        "embeddings":embeddings,

        "model":
        "ProtT5-xl-uniref50",

        "embedding_type":
        "mean_pooling"

    }


    torch.save(
        save,
        output
    )


    print(
        "Saved:",
        output
    )



# =========================================================

if __name__=="__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--fasta",
        required=True
    )


    parser.add_argument(
        "--output",
        required=True
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=1
    )


    parser.add_argument(
        "--seq_length",
        type=int,
        default=2000
    )


    args = parser.parse_args()


    extract_embedding(
        fasta=args.fasta,
        output=args.output,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )