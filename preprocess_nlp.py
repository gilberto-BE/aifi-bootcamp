from __future__ import unicode_literals, print_function, division
import torch
from io import open
import glob
import os
import unicodedata
import string


def findFiles(path):
    return glob.glob(path)


def unicodeToAscii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


def readLines(filename, all_letters):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line, all_letters) for line in lines]

def get_categories_lines():
    category_lines = {}
    all_categories = []

    for filename in findFiles('name_data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename, all_letters)
        category_lines[category] = lines
    return category_lines, all_categories


def letterToIndex(letter, all_letters):
    return all_letters.find(letter)


def letterToTensor(letter, n_letters, all_letters):
    """Fix issues with args"""
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter, all_letters)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

if __name__ == "__main__":
    print(findFiles('name_data/names/*txt'))
    all_letters = string.ascii_letters + ".,;'' "
    n_letters = len(all_letters)
    print()
    print('all_letters:', all_letters)
    print()
    print('n_letters:', n_letters)
    print()
    print('Swedish name: Bjön =>', unicodeToAscii('Björn', all_letters))
    #########################################
    category_lines, all_categories = get_categories_lines()
    n_categories = len(all_categories)
    n_categories

    print(category_lines['Italian'][:5])
    print(letterToTensor('J', n_letters))
    print()
    print(lineToTensor('Jones').size())
