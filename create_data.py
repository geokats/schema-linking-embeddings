import argparse
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


N_MAX = 6
agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
cond_ops = ['=', '>', '<', 'OP']

def wikisql_redable(sql, table):
    sql_readable = f"SELECT {agg_ops[sql['agg']]} {table['header'][sql['sel']]} FROM table"
    if sql['conds']:
        sql_readable += f" WHERE"
        for cond in sql['conds']:
            col_id, cond_op, val = cond
            sql_readable += f" {table['header'][col_id]} {cond_ops[cond_op]} {val}"

    return sql_readable

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input_file", required=True, help="Input WikiSQL .jsonl file")
    parser.add_argument("-t", "--table_file", required=True, help="Input WikiSQL .tables.jsonl file")

    return(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    #Load tables in dictionary
    tables = {}
    with open(args.table_file, 'r') as f:
        for line in f:
            table = json.loads(line)
            tables[table['id']] = table

    total_used = 0
    total_em = 0
    total_pm = 0

    with open(args.input_file, 'r') as f:
        for line in tqdm(f.readlines()):
            example = json.loads(line)
            table = tables[example['table_id']]
            header = table['header']


            nlq = example['question']
            tokens = nltk.word_tokenize(nlq.lower())
            sql = example['sql']
            sql_readable = wikisql_redable(sql, table)


            n_grams = []
            #Get all n-grams of length from N_MAX down to 1 (single words)
            for n in range(N_MAX+1, 0, -1):
                n_grams.extend([' '.join(x) for x in nltk.ngrams(tokens, n)])



            used_col_ids = []
            used_col_ids.append(sql['sel'])
            used_col_ids.extend([x[0] for x in sql['conds']])
            used_cols = [header[x].lower() for x in used_col_ids]

            # vectorizer = TfidfVectorizer()
            # vectorizer.fit(tokens + used_cols)



            total_used += len(used_cols)
            # col_vec = vectorizer.transform(used_cols)
            # print(col_vec)
            # print()
            # n_gram_vec = vectorizer.transform(n_grams)
            # print(n_gram_vec)
            # print()
            # sim = cosine_similarity(col_vec, n_gram_vec)
            # print(sim)

            links = []
            for col in used_cols:
                link = None
                best_sim = -1
                for n_gram in n_grams:
                    if n_gram == col:
                        # print("Exact match!")
                        link = n_gram
                        total_em += 1
                        continue
                    elif n_gram in col and link is None:
                        link = n_gram
                        total_pm += 1

                links.append(link)

            # print(example)
            print(f"NLQ:     {nlq}")
            # print(f"         {tokens}")
            # print(f"SQL:     {sql_readable}")
            # print(f"n-grams: {n_grams}")

            # print(f"Header:  {header}")
            print(f"Used:   {used_cols}")
            print(links)

            # break
    print(f"{total_em}/{total_used} ({round(total_em/total_used * 100, 2)}%) exact matches")
    print(f"{total_pm}/{total_used} ({round(total_pm/total_used * 100, 2)}%) partial matches")
