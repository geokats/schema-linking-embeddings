import pandas as pd

from EmbDI.edgelist import EdgeList
from EmbDI.utils import *
from EmbDI.graph import graph_generation
from EmbDI.sentence_generation_strategies import random_walks_generation
from EmbDI.embeddings import learn_embeddings

def create_table_emb(tdf, emb_file, emb_alg, emb_dim, temp_dir):
    """
    Use EmbDI to create table embeddings for a WikISQL table

        Parameters:
            tdf : The Dataframe of a WikiSQL table
    """
    edge_file = os.path.join(temp_dir, "tmp.edgelist")
    walks_file = os.path.join(temp_dir, "tmp.walks")
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


    model = learn_embeddings(emb_file, walks_file, True, emb_dim, 15,
                     training_algorithm=emb_alg,
                     learning_method='skipgram', sampling_factor=0.001
                    )
    return model

def wikisql_table_to_df(table):
    #We must remove all commas, otherwise EmbDI cannot read the edgefile
    header = [h.replace(',','') for h in table['header']]
    df = pd.DataFrame(columns=header, data=table['rows'])
    df = df.replace(',','', regex=True)
    return df