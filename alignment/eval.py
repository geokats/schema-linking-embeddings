#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2016-present, Facebook, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import io
import numpy as np
import argparse
from utils import *

parser = argparse.ArgumentParser(description='Evaluation of word alignment')
parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
parser.add_argument("--src_mat", type=str, default='', help="Load source alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--tgt_mat", type=str, default='', help="Load target alignment matrix. If none given, the aligment matrix is the identity.")
parser.add_argument("--dico_test", type=str, default='', help="test dictionary")
parser.add_argument("--maxload", type=int, default=200000)
parser.add_argument("--nomatch", action='store_true', help="no exact match in lexicon")
params = parser.parse_args()


###### SPECIFIC FUNCTIONS ######
# function specific to evaluation
# the rest of the functions are in utils.py

def load_transform(fname, d1=300, d2=300):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    R = np.zeros([d1, d2])
    for i, line in enumerate(fin):
        tokens = line.split(' ')
        R[i, :] = np.array(tokens[0:d2], dtype=float)
    return R


###### MAIN ######

print("Evaluation of alignment on %s" % params.dico_test)
if params.nomatch:
    print("running without exact string matches")

words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)

if params.tgt_mat != "":
    R_tgt = load_transform(params.tgt_mat)
    x_tgt = np.dot(x_tgt, R_tgt)
if params.src_mat != "":
    R_src = load_transform(params.src_mat)
    x_src = np.dot(x_src, R_src)

src2tgt, lexicon_size = load_lexicon(params.dico_test, words_src, words_tgt)

nnacc = compute_nn_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
cslsproc = compute_csls_accuracy(x_src, x_tgt, src2tgt, lexicon_size=lexicon_size)
print("NN = %.4f - CSLS = %.4f - Coverage = %.4f" % (nnacc, cslsproc, len(src2tgt) / lexicon_size))
