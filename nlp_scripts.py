from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
from rnn_nlp import RNN
from preprocess_nlp import (
    letterToIndex, 
    letterToTensor,
    get_categories_lines
 )
import torch

def clasify_names():
    all_letters = string.ascii_letters + ".,;''"
    n_letters = len(all_letters)
    input = letterToTensor('A', n_letters)

if __name__ == "__main__":
    all_letters = string.ascii_letters + ".,;''"
    n_letters = len(all_letters)
    input = letterToTensor('A', n_letters)
    n_hidden = 256
    category_lines, all_categories = get_categories_lines()
    n_categories = len(all_categories)
    rnn = RNN(n_letters, n_hidden, n_categories)
    hidden = torch.zeros(1, n_hidden)
    output, next_hidden = rnn(input, hidden)
    print(output)