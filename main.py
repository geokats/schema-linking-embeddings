import json
import argparse
import random
import os
import sys
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tqdm.auto import tqdm

from alignment.utils import load_vectors, idx, select_vectors_from_pairs
from alignment.align import align_embeddings

from util import create_table_emb, wikisql_table_to_df

TEMP_DIR = "./tmp/"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-t", "--tables_file", required=True, help="Input WikiSQL .tables.jsonl file")
    parser.add_argument("-e", "--embeddings_file", required=True, help="Pre-trained embeddings file")
    parser.add_argument("-d", "--embeddings_dimension", type=int, required=True, help="Dimension of pre-trained embeddings")
    parser.add_argument("-a", "--embeddings_algorithm", choices=['word2vec', 'fasttext'], required=True, help="Algorithm to use for training local embeddings")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to save the table embeddings and alignment matrices")

    return(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()

    #Create output directory
    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        pass
    #Create temporary directory for storing auxilary files
    try:
        os.makedirs(TEMP_DIR)
    except FileExistsError:
        pass

    #Load the pre-trained word embeddings
    #NOTE: In our case the pre-trained embeddings are the source embeddings, while
    #      the table embeddings are the target embeddings, and we want to find
    #      a mapping from the table embeddings space to the pre-trained space
    kv_pre = KeyedVectors.load_word2vec_format(args.embeddings_file)
    vec_pre = kv_pre.vectors
    words_pre_set = set(kv_pre.key_to_index.keys())
    idx_pre = kv_pre.key_to_index

    with open(args.tables_file, 'r') as f:
        #Read a table
        for line in tqdm(f.readlines()):
            #Load table
            table = json.loads(line)
            table_id = table['id']
            tdf = wikisql_table_to_df(table)

            #File to save new table embeddings
            emb_out_file = os.path.join(args.output_dir, f"{table_id}.emb")
            #File to save alignment matrix
            matrix_out_file = os.path.join(args.output_dir, f"{table_id}.R.npy")

            #Create table embeddings
            model = create_table_emb(tdf, emb_out_file, args.embeddings_algorithm, args.embeddings_dimension, TEMP_DIR)

            #Load table embeddings
            kv_tab = model.wv
            vec_tab = kv_tab.vectors
            words_tab_set = set(kv_tab.key_to_index.keys())
            idx_tab = kv_tab.key_to_index

            #Find anchor words
            anchors = words_tab_set & words_pre_set #Common words are the intersection
            pairs = [(idx_pre[w], idx_tab[w]) for w in anchors]

            #Align pre-trained vectors to the table embedding space
            R = align_embeddings(vec_pre, vec_tab, pairs)

            #Save alignment matrix
            np.save(matrix_out_file, R)
