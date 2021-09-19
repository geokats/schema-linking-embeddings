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

def get_col_matches(tokens, tdf, kv, threshold=0.5):
    column_tokens = []
    for col_id, col in enumerate(tdf.columns):
        cur_col = [f"cid__{col_id}"] + col.split(' ')

        cur_col = [x for x in cur_col if kv.has_index_for(x)]

        column_tokens.append(cur_col)

    matches = []

    for token in tokens:
        if not kv.has_index_for(token):
            continue

        min_dist = 1000
        best_col = None
        for col_id, col in enumerate(column_tokens):
            dists = kv.distances(token, col)
            dist = dists.mean()
            if best_col is None or dist < min_dist:
                min_dist = dist
                best_col = col_id

#         print(f"{token} -> {table['header'][best_col]} ({min_dist})")
        matches.append((token, best_col, min_dist))

    #Keep all matches above threshold or atleast the best match
    best_matches = [(token, col) for token, col, dist in matches if dist < threshold]
    if len(best_matches) == 0:
        min_dist = min([dist for token, col, dist in matches])
        best_matches = [(token, col) for token, col, dist in matches if dist == min_dist]

    return best_matches

def get_row_matches(tokens, tdf, kv, threshold=0.5):
    row_tokens = []
    for i, row in tdf.iterrows():
        cur_row = [f"idx__{i}"]
        for val in row.values:
            cur_row.extend(val.split(' '))

        cur_row = [x for x in cur_row if kv.has_index_for(x)]

        row_tokens.append(cur_row)

    matches = []

    for token in tokens:
        if not kv.has_index_for(token):
            continue

        min_dist = 1000
        best_row = None
        for row_id, row in enumerate(row_tokens):
            dists = kv.distances(token, row)
            dist = dists.mean()
            if best_row is None or dist < min_dist:
                min_dist = dist
                best_row = row_id

#         print(f"{token} -> {best_row} ({min_dist})")
        if min_dist < threshold:
            matches.append((token, best_row))

    return matches

def add_aligned_vectors(kv, tokens, vec_pre, idx_pre, R):
    add_words = []
    add_vecs = []

    for token in tokens:
        if not kv.has_index_for(token):
            add_words.append(token)
            add_vecs.append(vec_pre[idx_pre[token]])

    aligned_vecs = vector_align(np.array(add_vecs), R)
    kv.add_vectors(add_words, aligned_vecs, replace=False)

    return kv

def get_stats(matches, gt_matches):
    matched_pred = [False for _ in matches]
    matched_gt = [False for _ in gt_matches]

    for gt_idx, (gt_token, gt_col_id) in enumerate(gt_matches):
        for pred_idx, (token, col_id) in enumerate(matches):
            if matched_pred[pred_idx]:
                continue
            if token.lower() in gt_token and col_id == gt_col_id:
                matched_pred[pred_idx] = True
                matched_gt[gt_idx] = True
                break

    tp = sum(matched_gt)
    fp = len(matched_pred) - sum(matched_pred)
    fn = len(matched_gt) - sum(matched_gt)

    return tp, fp, fn

def get_prec(stats):
    try:
        tab_prec = stats['tab_tp']/(stats['tab_tp']+stats['tab_fp'])
    except ZeroDivisionError:
        tab_prec = -1

    try:
        pre_prec = stats['pre_tp']/(stats['pre_tp']+stats['pre_fp'])
    except ZeroDivisionError:
        pre_prec = -1

    return tab_prec, pre_prec

def get_rec(stats):
    try:
        tab_rec = stats['tab_tp']/(stats['tab_tp']+stats['tab_fn'])
    except ZeroDivisionError:
        tab_rec = -1

    try:
        pre_rec = stats['pre_tp']/(stats['pre_tp']+stats['pre_fn'])
    except ZeroDivisionError:
        pre_rec = -1

    return tab_rec, pre_rec
