import argparse
import json
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from util import wikisql_redable

N_MAX = 7

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def find_name_link(col_name, n_grams, n_grams_tok):
    link = None
    best_sim = 0
    col_tok = nltk.word_tokenize(col_name)

    for n_gram, n_gram_tok in zip(n_grams, n_grams_tok):
        if n_gram == col_name:
            link = n_gram
            break
        else:
            sim = jaccard_similarity(col_tok, n_gram_tok)
            if sim > best_sim:
                best_sim = sim
                link = n_gram

    return link

def print_example():
    print(nlq)
    print(sql_readable)
    print(example['schema_links'])
    print()

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-i", "--input_file", required=True, help="Input WikiSQL .jsonl file")
    parser.add_argument("-t", "--table_file", required=True, help="Input WikiSQL .tables.jsonl file")
    parser.add_argument("-o", "--output_file", required=True, help="Output file to save the updated examples with schema links")

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

    stats = {
        "examples" : 0, #Total number of examples seen
        "coverage" : 0, #Number of examples with found SELECT link and either a name or value link for all WHERE columns

        "sel_cols" : 0, #Total number of SELECT columns in the dataset
        "sel_em" : 0, #Number of SELECT columns matched exactly
        "sel_pm" : 0, #Number of SELECT columns matched partially

        "whr_cols" : 0, #Total number of WHERE columns in the dataset
        "whr_em" : 0, #Number of WHERE columns matched exactly
        "whr_pm" : 0, #Number of WHERE columns matched partially

        "whr_vals" : 0, #Total number of WHERE values in the dataset
        "val_em" : 0, #Number of WHERE values matched exactly
        "val_pm" : 0, #Number of WHERE values matched partially
    }

    lemmatizer = WordNetLemmatizer()

    with open(args.input_file, 'r') as inp_f, open(args.output_file, 'w') as out_f:
        for line in tqdm(inp_f.readlines()):
            example = json.loads(line)
            example['schema_links'] = {}
            example['schema_links']['where_name'] = []
            example['schema_links']['where_value'] = []

            #Get the table of the example
            table = tables[example['table_id']]
            header = table['header']

            #Get NLQ and SQL of the example
            nlq = example['question']
            sql = example['sql']
            sql_readable = wikisql_redable(sql, table)

            #Tokenize the NLQ
            tokens = nltk.word_tokenize(nlq.lower())

            #Get all n-grams in the NLQ of length from N_MAX down to 1 (single words)
            n_grams_tok = []
            for n in range(N_MAX+1, 0, -1):
                n_grams_tok.extend(nltk.ngrams(tokens, n))
            n_grams = [' '.join(x) for x in n_grams_tok]

            #Find a name link for the SELECT column
            sel_col_name = header[sql['sel']].lower()
            link = find_name_link(sel_col_name, n_grams, n_grams_tok)

            #Save SELECT column name link
            example['schema_links']['select_name'] = link

            #Update statistics
            stats['sel_cols'] += 1
            if link != None:
                if link == sel_col_name:
                    stats['sel_em'] += 1
                else:
                    stats['sel_pm'] += 1

            #Find a name link for the WHERE columns
            where_col_names = [header[x[0]].lower() for x in sql['conds']]
            for where_col_name in where_col_names:
                link = find_name_link(where_col_name, n_grams, n_grams_tok)

                #Save WHERE column name link
                example['schema_links']['where_name'].append(link)

                #Update statistics
                stats['whr_cols'] += 1
                if link != None:
                    if link == where_col_name:
                        stats['whr_em'] += 1
                    else:
                        stats['whr_pm'] += 1

            #Find a value link for the WHERE columns
            values = [str(x[2]).lower() for x in sql['conds']]
            for value in values:
                link = find_name_link(value, n_grams, n_grams_tok)

                #Save WHERE column value link
                example['schema_links']['where_value'].append(link)

                #Update statistics
                stats['whr_vals'] += 1
                if link != None:
                    if link == value:
                        stats['val_em'] += 1
                    else:
                        stats['val_pm'] += 1

            #Save example and update statistics
            stats['examples'] += 1
            if example['schema_links']['select_name'] != None:
                good_example = True
                for name_link, val_link in zip(example['schema_links']['where_name'], example['schema_links']['where_value']):
                    if name_link == None and val_link == None:
                        good_example = False

                if good_example:
                    stats['coverage'] += 1
                    out_f.write(json.dumps(example))
                    out_f.write('\n')

    #Print statistics
    print(f"{stats['coverage']}/{stats['examples']} ({round(stats['coverage']/stats['examples'] * 100, 2)}%) examples covered\n")

    print(f"{stats['sel_em']}/{stats['sel_cols']} ({round(stats['sel_em']/stats['sel_cols'] * 100, 2)}%) SELECT exact matches")
    print(f"{stats['sel_pm']}/{stats['sel_cols']} ({round(stats['sel_pm']/stats['sel_cols'] * 100, 2)}%) SELECT partial matches")
    print(f"{stats['sel_em']+stats['sel_pm']}/{stats['sel_cols']} ({round((stats['sel_em']+stats['sel_pm'])/stats['sel_cols'] * 100, 2)}%) SELECT total coverage\n")

    print(f"{stats['whr_em']}/{stats['whr_cols']} ({round(stats['whr_em']/stats['whr_cols'] * 100, 2)}%) WHERE column exact matches")
    print(f"{stats['whr_pm']}/{stats['whr_cols']} ({round(stats['whr_pm']/stats['whr_cols'] * 100, 2)}%) WHERE column partial matches")
    print(f"{stats['whr_em']+stats['whr_pm']}/{stats['whr_cols']} ({round((stats['whr_em']+stats['whr_pm'])/stats['whr_cols'] * 100, 2)}%) WHERE column total coverage\n")

    print(f"{stats['val_em']}/{stats['whr_vals']} ({round(stats['val_em']/stats['whr_vals'] * 100, 2)}%) WHERE value exact matches")
    print(f"{stats['val_pm']}/{stats['whr_vals']} ({round(stats['val_pm']/stats['whr_vals'] * 100, 2)}%) WHERE value partial matches")
    print(f"{stats['val_em']+stats['val_pm']}/{stats['whr_vals']} ({round((stats['val_em']+stats['val_pm'])/stats['whr_vals'] * 100, 2)}%) WHERE value total coverage")
