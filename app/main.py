from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from gensim.models import KeyedVectors
import os
import json
import random
import numpy as np
import nltk
import sys

sys.path.insert(1, '../')
from util import get_row_matches, get_col_matches, add_aligned_vectors, wikisql_table_to_df


SLEMB_DIR = "./slemb"
TABLES_FILE = "../wikisql/data/dev.tables.jsonl"
EMB_FILE = "../embeddings/wiki-news-300d-1M.vec"
COL_THRESHOLD = 0.5
ROW_THRESHOLD = 0.5

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

#Load pre-trained word embeddings
kv_pre = KeyedVectors.load_word2vec_format(EMB_FILE)
vec_pre = kv_pre.vectors
words_pre_set = set(kv_pre.key_to_index.keys())
idx_pre = kv_pre.key_to_index

#Load available table id embeddings
files = os.listdir(SLEMB_DIR)
table_ids = set()
for file_name in files:
    if os.path.isfile(os.path.join(SLEMB_DIR, file_name)):
        file_name = file_name.rstrip(".emb")
        file_name = file_name.rstrip(".R.npy")
        table_ids.add(file_name)

#Load tables
tables = {}
with open(TABLES_FILE, 'r') as f:
    for line in f:
        obj = json.loads(line)
        if obj['id'] in table_ids:
            tables[obj['id']] = obj
        if len(tables) == len(table_ids):
            break

current_id = random.choice(list(table_ids))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "table_ids": table_ids, "table": tables[current_id], "query" : "", "results" : {}})

@app.post("/", response_class=HTMLResponse)
async def root_post(request: Request, query: str = Form(...)):
    #Load table
    tdf = wikisql_table_to_df(tables[current_id])
    #Load table embeddings and R matrix
    kv_tab = KeyedVectors.load_word2vec_format(os.path.join(SLEMB_DIR, f"{current_id}.emb"))
    R = np.load(os.path.join(SLEMB_DIR, f"{current_id}.R.npy"))
    #Tokenize NLQ
    tokens = nltk.word_tokenize(query)
    #Align NLQ tokens to local embeddings
    kv_tab = add_aligned_vectors(kv_tab, tokens, vec_pre, idx_pre, R)
    #Get column matches
    col_matches = get_col_matches(tokens, tdf, kv_tab, threshold=COL_THRESHOLD)
    #Get row matches
    row_matches = get_row_matches(tokens, tdf, kv_tab, threshold=ROW_THRESHOLD)

    results = {'col_matches' : col_matches, 'row_matches' : row_matches}

    return templates.TemplateResponse("index.html", {"request": request, "table_ids": table_ids, "table": tables[current_id], "query" : query, "results" : results})

@app.get("/{table_id}", response_class=RedirectResponse)
def change_table(request: Request, table_id: str):
    global current_id
    current_id = table_id
    return RedirectResponse("/")
