#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function
import os, argparse, logging, sys
import string
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.decomposition import PCA #, IncrementalPCA
from pandas import Series, DataFrame
import pandas as pd


logger = logging.getLogger('salan')
log_filename = 'log.txt'

def read_array(ifname):
    if not os.path.isfile(ifname):
        logger.warn('Array file does not exist: %s' % ifname)
        print('WARNING: plotutils - Array file does not exist: %s' % ifname)
        return ([], [])
    with open(ifname, 'rU') as ifile:
        labels = ifile.readline().strip().split('\t')[1:]
        matrix = []
        for line in ifile:
            matrix.extend(list(np.float32(x) for x in [line.strip().split('\t')[1:]]))
    return (labels, np.array(matrix))

def read_dataframe(ifname, index_col=0, multi=False):
    if not os.path.isfile(ifname):
        logger.warn('Array file does not exist: %s' % ifname)
        print('WARNING: plotutils - Array file does not exist: %s' % ifname)
        return None
    if multi:
        return pd.read_table(ifname, index_col=[0, 1], header=[0, 1])
    else:
        return pd.read_table(ifname, index_col=index_col)


def plotpca_mat(labels, matrix, n_components = 2):
    from IPython import embed
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(matrix)
    for x, y in X_pca:
        plt.scatter(x, y)
    plt.show()
    embed()
    return X_pca

def pca(df, n_components = 2, ofname_prefix=''):
    from IPython import embed
    if not ofname_prefix:
        ofname_prefix = 'out-pca'
    component_fname = ofname_prefix + '-component-' + str(n_components) + '.txt'
    score_fname = ofname_prefix + '-score-comp-' + str(n_components) + '.txt'
    transdata_fname = ofname_prefix + '-transdata-' + str(n_components) + '.txt'
    xwriter = pd.ExcelWriter(ofname_prefix+'-'+str(n_components)+'.xlsx', engine='xlsxwriter')
    pca = PCA(n_components=n_components)
    #get locations for last non-data column and row
    if 'TOTAL' in df.index.get_values():
        total_loc = df.index.get_loc('TOTAL') #presume only one, since n-grams are lower case
    else:
        total_loc = 0
    last_demo_col_loc = df.columns.get_loc('Frequency')
    sentiment_cols = False
    if 'D.Mean.Sum' in df.columns: #Sentiment Analysis included
        last_demo_col_loc = df.columns.get_loc('D.Mean.Sum')    
        sentiment_cols = df.columns[df.columns.get_loc('Frequency')+1:last_demo_col_loc+1]
    df_data = df.drop(df.columns[0:last_demo_col_loc+1], axis=1)
    participant_word_totals = df_data.iloc[total_loc]
    df_data = df_data[total_loc+1:]
    #calculate relative frequency
    df_freq = df_data / participant_word_totals #spot checked correct 
    X_pca = pca.fit_transform(df_freq)
    #transformed data
    if sentiment_cols:
        df_tran = df[sentiment_cols]
        df_tran = df_tran[total_loc+1:]
    else:
        df_tran = DataFrame()
    df_tran = pd.concat([df_tran, DataFrame(X_pca, index=df_freq.index)], axis=1)
    df_tran.to_csv(transdata_fname, sep='\t', index_label='NGrams')
    df_tran.to_excel(xwriter, index_label='NGrams', sheet_name='Transformed')
    #components
    df_pca = df[0:total_loc]
    df_pca = df_pca.drop(df.columns[0:last_demo_col_loc+1], axis=1)
    df_pca = df_pca.append(DataFrame(pca.components_, columns=df_pca.columns))
    df_pca.to_csv(component_fname, sep='\t', index_label='Dim')
    df_pca.to_excel(xwriter, index_label='Dim', sheet_name='Components')
    #score
    s_score = Series(pca.score_samples(df_freq), index=df_freq.index)
    s_score.to_csv(score_fname, sep='\t')
    df_score = DataFrame(s_score)
    df_score.to_excel(xwriter, sheet_name='Score')
    DataFrame(pca.explained_variance_ratio_).to_excel(xwriter, sheet_name='Explain Var')
    xwriter.save()
    #df_score.sort(ascending=False)
    #plot
    #for x, y in X_pca:
    #    plt.scatter(x, y)
    #plt.show()
    embed()
    return X_pca

def main():
    from IPython import embed
    args = init_args()
    if args.pca:
        df1 = read_dataframe(args.pca, multi=args.multi)
        if args.multi:
            pca(df1.drop(0,axis=1).drop(1,axis=1))
        else:
            pca(df1, n_components=args.n_components)
    if args.embed:
        embed()

def init_args(parser='', scriptpath=''):
    if not parser:
        parser = argparse.ArgumentParser(description='Run and plot MDS from tab-delimited grid file.')
    if not scriptpath:
        scriptpath = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--embed', dest='embed', action="store_true", help='Bring up an IPython prompt after running.')
    parser.add_argument('--pca', dest='pca', default="", help='Run principal component analysis (PCA) on grid file (ie, wide).')
    parser.add_argument('--components', dest='n_components', type=int, default=2, help='Number of components.')
    parser.add_argument('--multi2', dest='multi', action="store_true", help='Use first two (2) columns and rows as multi-index (pca only).')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)    
    #logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()
