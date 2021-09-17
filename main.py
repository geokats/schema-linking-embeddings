import json
import argparse
import random
import pandas as pd

from EmbDI.edgelist import EdgeList
from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
from EmbDI.embeddings import learn_embeddings

from alignment.utils import load_vectors, idx, select_vectors_from_pairs
from alignment.align import align_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input_file", required=True, help="Input WikiSQL .tables.jsonl file")

    return(parser.parse_args())

def wikisql_table_to_df(table):
    return pd.DataFrame(columns=table['header'], data=table['rows'])

def create_table_emb(tdf):
    """
    Use EmbDI to create table embeddings for a WikISQL table

        Parameters:
            tdf : The Dataframe of a WikiSQL table
    """
    edgefile = "example.edgelist"
    prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']
    info = None

    edgelist = EdgeList(tdf, edgefile, prefixes, info, flatten=True)

    # Default parameters
    configuration = {
        'walks_strategy': 'basic',
        'walks_file' : 'example.walks',
        'flatten': 'all',
        'input_file': 'example.edgelist',
        'n_sentences': 'default',
        'sentence_length': 10,
        'write_walks': True,
        'intersection': False,
        'backtrack': True,
        'output_file': 'example',
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


    learn_embeddings("example.emb", "example.walks", True, 300, 15,
                     training_algorithm='fasttext',
                     learning_method='skipgram', sampling_factor=0.001
                    )


if __name__ == '__main__':
    args = parse_args()

    #Load the pre-trained word embeddings
    #NOTE: In our case the pre-trained embeddings are the source embeddings, while
    #      the table embeddings are the target embeddings, and we want to find
    #      a mapping from the table embeddings space to the pre-trained space
    words_pre, vec_pre = load_vectors("embeddings/wiki-news-300d-1M.vec", maxload=10000)
    words_pre_set = set(words_pre)
    idx_pre = idx(words_pre)

    with open(args.input_file, 'r') as f:
        #Read a table
        # line = random.choice(f.readlines())
        line = f.readlines()[3]
        table = json.loads(line)
        tdf = wikisql_table_to_df(table)
        print(tdf)

        #Create table embeddings
        create_table_emb(tdf)

        #Load table embeddings
        words_tab, vec_tab = load_vectors("example.emb", maxload=-1)
        words_tab_set = set(words_tab)
        idx_tab = idx(words_tab)

        #Find anchor words
        anchors = words_tab_set & words_pre_set #Common words are the intersection
        print(f"Anchor words: {anchors}")
        pairs = [(idx_pre[w], idx_tab[w]) for w in anchors]

        #Align pre-trained vectors to the table embedding space
        print("Aligning...")
        R = align_embeddings(vec_pre, vec_tab, pairs)
        print(R )

        #Save alignment matrix
        np.save("example-R.npy", R)
