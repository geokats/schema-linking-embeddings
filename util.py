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

    #Create edgelist
    EdgeList(tdf, edge_file, prefixes, info, flatten=True)
    prefixes, edgelist = read_edgelist(configuration['input_file'])

    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    random_walks_generation(configuration, graph)

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

def wikisql_redable(sql, table):
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    sql_readable = f"SELECT {agg_ops[sql['agg']]} {table['header'][sql['sel']]} FROM table"
    if sql['conds']:
        sql_readable += f" WHERE"
        for cond in sql['conds']:
            col_id, cond_op, val = cond
            sql_readable += f" {table['header'][col_id]} {cond_ops[cond_op]} {val}"

    return sql_readable

def vector_align(x, R):
    x_new = np.dot(x, R.T)
    x_new /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    return x_new

def get_col_matches(tokens, tdf, model, threshold=0.5):
    column_tokens = []
    for i, col in enumerate(tdf.columns):
    #     column_tokens.append([f"cid__{i}"] + nltk.word_tokenize(col))
        column_tokens.append([f"cid__{i}"] + col.split(' '))

    matches = []

    for token in tokens:
        min_dist = 1000
        best_col = None
        for col_id, col in enumerate(column_tokens):
            dists = model.wv.distances(token, col)
            dist = dists.mean()
            if best_col is None or dist < min_dist:
                min_dist = dist
                best_col = col_id

#         print(f"{token} -> {table['header'][best_col]} ({min_dist})")
        if min_dist < threshold:
            matches.append((token, best_col))

    return matches

def get_row_matches(tokens, tdf, model, threshold=0.5):
    row_tokens = []
    for i, row in tdf.iterrows():
        cur_row = [f"idx__{i}"]
        for val in row.values:
            cur_row.extend(val.split(' '))

        row_tokens.append(cur_row)

    matches = []

    for token in tokens:
        min_dist = 1000
        best_row = None
        for row_id, row in enumerate(row_tokens):
            dists = model.wv.distances(token, row)
            dist = dists.mean()
            if best_row is None or dist < min_dist:
                min_dist = dist
                best_row = row_id

#         print(f"{token} -> {best_row} ({min_dist})")
        if min_dist < threshold:
            matches.append((token, best_row))

    return matches
