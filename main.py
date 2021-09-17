import json
import argparse
import random
import pandas as pd

from EmbDI.edgelist import EdgeList
from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
from EmbDI.embeddings import learn_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input_file", required=True, help="Input WikiSQL .tables.jsonl file")

    return(parser.parse_args())

def wikisql_table_to_df(table):
    return pd.DataFrame(columns=table['header'], data=table['rows'])

if __name__ == '__main__':
    args = parse_args()

    with open(args.input_file, 'r') as f:
        #Read a table
        # line = random.choice(f.readlines())
        line = f.readlines()[3]
        table = json.loads(line)
        tdf = wikisql_table_to_df(table)
        print(tdf)

        edgefile = "example.edgelist"
        prefixes = ['3#__tn', '3$__tt', '5$__idx', '1$__cid']
        info = None

        edgelist = EdgeList(tdf, edgefile, prefixes, info, flatten=True)
        edge_dict = edgelist.convert_to_dict()
        # for k,v in edge_dict.items():
        #     print(f"{k}: {v}")

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

        # df = pd.read_csv('pipeline/datasets/small_example.csv')
        #
        prefixes, edgelist = read_edgelist(configuration['input_file'])

        graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
        if configuration['n_sentences'] == 'default':
            #  Compute the number of sentences according to the rule of thumb.
            configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
        walks = random_walks_generation(configuration, graph)


        learn_embeddings("example.emb", "pipeline/walks/example.walks", True, 100, 15, training_algorithm='word2vec',
                             learning_method='skipgram', sampling_factor=0.001)
