import json
import argparse
import random
import pandas as pd

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
        line = random.choice(f.readlines())
        table = json.loads(line)
        tdf = wikisql_table_to_df(table)
        print(tdf)
