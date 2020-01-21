# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 10:05:22 2018

@author: gilbe
"""

import os

"""
File to contain global variables / function which are useful for all aspects of the project.
"""
"""This file contains variables and functions
 which are helpful to all aspects of the package
"""

PROJECT_ROOT_DIR = os.path.join(os.path.dirname('__file__'), '.')
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'data')

if __name__ == '__main__':
    print(PROJECT_DATA_DIR)