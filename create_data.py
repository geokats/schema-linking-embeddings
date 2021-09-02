import argparse
import json

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

    with open(args.input_file, 'r') as f:
        for line in f:
            example = json.loads(line)
            table = tables[example['table_id']]

            print(example)

            nlq = example['question']
            sql = example['sql']
            sql_readable = wikisql_redable(sql, table)
            print(f"NLQ:    {nlq}")
            print(f"SQL:    {sql_readable}")

            header = table['header']
            print(f"Header: {header}")

            used_col_ids = []
            used_col_ids.append(sql['sel'])
            used_col_ids.extend([x[0] for x in sql['conds']])
            used_cols = [header[x] for x in used_col_ids]

            print(f"Used:   {used_cols}")

            break
