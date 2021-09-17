import json
import argparse
import random
import os
import pandas as pd
from tqdm.auto import tqdm

from EmbDI.edgelist import EdgeList
from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
from EmbDI.embeddings import learn_embeddings

from alignment.utils import load_vectors, idx, select_vectors_from_pairs
from alignment.align import align_embeddings

TEMP_DIR = "./tmp/"

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-t", "--tables_file", required=True, help="Input WikiSQL .tables.jsonl file")
    parser.add_argument("-e", "--embeddings_file", required=True, help="Pre-trained embeddings file")
    parser.add_argument("-o", "--output_dir", required=True, help="Output directory to save the table embeddings and alignment matrices")


    return(parser.parse_args())

def wikisql_table_to_df(table):
    df = pd.DataFrame(columns=table['header'], data=table['rows'])
    df = df.replace(',','', regex=True)
    return df

def create_table_emb(tdf, emb_file):
    """
    Use EmbDI to create table embeddings for a WikISQL table

        Parameters:
            tdf : The Dataframe of a WikiSQL table
    """
    edge_file = os.path.join(TEMP_DIR, "tmp.edgelist")
    walks_file = os.path.join(TEMP_DIR, "tmp.walks")
    prefixes = ['3$__tn', '3$__tt', '5$__idx', '1$__cid']
    info = None

    edgelist = EdgeList(tdf, edge_file, prefixes, info, flatten=True)

    # Default parameters
    configuration = {
        'walks_strategy': 'basic',
        'walks_file' : walks_file,
        'flatten': 'all',
        'input_file': edge_file,
        'n_sentences': 'default',
        'sentence_length': 10,
        'write_walks': True,
        'intersection': False,
        'backtrack': True,
        # 'output_file': id,
        'repl_numbers': False,
        'repl_strings': False,
        'follow_replacement': False,
        'mlflow': False
    }

    prefixes, edgelist = read_edgelist(configuration['input_file'])

    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    walks = random_walks_generation(configuration, graph)


    learn_embeddings(emb_file, walks_file, True, 300, 15,
                     training_algorithm='fasttext',
                     learning_method='skipgram', sampling_factor=0.001
                    )


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
    words_pre, vec_pre = load_vectors(args.embeddings_file, maxload=10000)
    words_pre_set = set(words_pre)
    idx_pre = idx(words_pre)

    with open(args.tables_file, 'r') as f:
        #Read a table
        for line in tqdm(f.readlines()):
            #Load table
            table = json.loads(line)
            table_id = table['id']
            tdf = wikisql_table_to_df(table)
            print(tdf)

            #File to save new table embeddings
            emb_out_file = os.path.join(args.output_dir, f"{table_id}.emb")
            #File to save alignment matrix
            matrix_out_file = os.path.join(args.output_dir, f"{table_id}.R.npy")

            #Create table embeddings
            create_table_emb(tdf, emb_out_file)

            #Load table embeddings
            words_tab, vec_tab = load_vectors(emb_out_file, maxload=-1)
            words_tab_set = set(words_tab)
            idx_tab = idx(words_tab)

            #Find anchor words
            anchors = words_tab_set & words_pre_set #Common words are the intersection
            pairs = [(idx_pre[w], idx_tab[w]) for w in anchors]

            #Align pre-trained vectors to the table embedding space
            R = align_embeddings(vec_pre, vec_tab, pairs)

            #Save alignment matrix
            np.save(matrix_out_file, R)
            break
