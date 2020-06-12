import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")


# import lightgbm as lgb
import csv
import copy
import numpy as np
import pandas as pd  # only used to return a dataframe
import math
import neurom as nm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from graphviz import Digraph
import datetime
import os, errno, sys, shutil
from collections import OrderedDict
import matplotlib.ticker as ticker
import pylab
import time
import progressbar
import itertools as it
from sklearn.manifold import TSNE
from sklearn import preprocessing
from sklearn.neighbors.kde import KernelDensity
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, r2_score, classification_report, confusion_matrix, classification_report
from sklearn.model_selection import ShuffleSplit, learning_curve
import functools
import sys
from xgboost import XGBClassifier
from .settings import *
import heapq
# import h2o
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.spatial import distance
import ast
import random
from tqdm import tqdm
import pickle
import tensorflow as tf
from sys import platform
import gc
from scipy.spatial import ConvexHull


# pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 160)
np.set_printoptions(linewidth=10000)

# pd.set_option('display.height', 1000)
# pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

########################################################################################################################
def list_ancestors(edges):
    """
    Take edge list of a rooted tree as a numpy array with shape (E, 2),
    child nodes in edges[:, 0], parent nodes in edges[:, 1]
    Return pandas dataframe of all descendant/ancestor node pairs

    Ex:
        df = pd.DataFrame({'child': [200, 201, 300, 301, 302, 400],
                           'parent': [100, 100, 200, 200, 201, 300]})

        df
           child  parent
        0    200     100
        1    201     100
        2    300     200
        3    301     200
        4    302     201
        5    400     300

        list_ancestors(df.values)

        returns

            descendant  ancestor
        0          200       100
        1          201       100
        2          300       200
        3          300       100
        4          301       200
        5          301       100
        6          302       201
        7          302       100
        8          400       300
        9          400       200
        10         400       100
    """
    ancestors = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]-1),
                               ar[:, 1:].flatten()])
    return pd.DataFrame(np.concatenate(ancestors),
                        columns=['descendant', 'ancestor'])


def trace_nodes(edges):
    """
    Take edge list of a rooted tree as a numpy array with shape (E, 2),
    child nodes in edges[:, 0], parent nodes in edges[:, 1]
    Yield numpy array with cross-section of tree and associated
    ancestor nodes

    Ex:
        df = pd.DataFrame({'child': [200, 201, 300, 301, 302, 400],
                           'parent': [100, 100, 200, 200, 201, 300]})

        df
           child  parent
        0    200     100
        1    201     100
        2    300     200
        3    301     200
        4    302     201
        5    400     300

        trace_nodes(df.values)

        yields

        array([[200, 100],
               [201, 100]])

        array([[300, 200, 100],
               [301, 200, 100],
               [302, 201, 100]])

        array([[400, 300, 200, 100]])
    """
    # Top layer
    mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
    gen_branches = edges[~mask]   # branch of parent without further ancestor
    edges = edges[mask]   # branch of otherwise
    yield gen_branches    # generate result

    # Successor layers
    while edges.size != 0:
        mask = np.in1d(edges[:, 1], edges[:, 0])    # parent with/without further ancestor
        next_gen = edges[~mask]   # branch of parent without further ancestor
        gen_branches = numpy_col_inner_many_to_one_join(next_gen, gen_branches)  # connect with further ancestors
        edges = edges[mask]   # branch of otherwise
        yield gen_branches    # generate result


def numpy_col_inner_many_to_one_join(ar1, ar2):
    """
    Take two 2-d numpy arrays ar1 and ar2,
    with no duplicate values in first column of ar2
    Return inner join of ar1 and ar2 on
    last column of ar1, first column of ar2

    Ex:

        ar1 = np.array([[1,  2,  3],
                        [4,  5,  3],
                        [6,  7,  8],
                        [9, 10, 11]])

        ar2 = np.array([[ 1,  2],
                        [ 3,  4],
                        [ 5,  6],
                        [ 7,  8],
                        [ 9, 10],
                        [11, 12]])

        numpy_col_inner_many_to_one_join(ar1, ar2)

        returns

        array([[ 1,  2,  3,  4],
               [ 4,  5,  3,  4],
               [ 9, 10, 11, 12]])
    """
    # Select connectable rows of ar1 and ar2 (ie. ar1 last_col = ar2 first col)
    ar1 = ar1[np.in1d(ar1[:, -1], ar2[:, 0])]   # error occurred if ar1 is empty.
    ar2 = ar2[np.in1d(ar2[:, 0], ar1[:, -1])]

    # if int >= 0, else otherwise
    if 'int' in ar1.dtype.name and ar1[:, -1].min() >= 0:
        bins = np.bincount(ar1[:, -1])
        counts = bins[bins.nonzero()[0]]
    else:
        # order of np is "-int -> 0 -> +int -> other type"
        counts = np.unique(ar1[:, -1], False, False, True)[1]

    # Reorder array with np's order rule
    left = ar1[ar1[:, -1].argsort()]
    right = ar2[ar2[:, 0].argsort()]

    # Connect the rows of ar1 & ar2
    return np.concatenate([left[:, :-1],
                           right[np.repeat(np.arange(right.shape[0]),
                                           counts)]], 1)

########################################################################################################################
def ancestors_and_path(df, child_col='ID', parent_col='PARENT_ID'):
    df_anc = df[[child_col, parent_col]]

    edges = df_anc.values

    ancestors = []
    path = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1] - 1),
                               ar[:, 1:].flatten()])
        path.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]),
                          ar[:, :].flatten()])

    return pd.DataFrame(np.concatenate(ancestors),columns=['descendant', 'ancestor']), \
           pd.DataFrame(np.concatenate(path),columns=['descendant', 'path'])


########################################################################################################################
def neuron_ancestors_and_path(df, child_col='ID', parent_col='PARENT_ID'):
    df_anc = df[[child_col, parent_col]]

    ### Need to drop row that "PARENT_ID = -1" first
    df_anc = df_anc[df_anc[parent_col] != -1]

    edges = df_anc.values

    ancestors = []
    path = []
    for ar in trace_nodes(edges):
        ancestors.append(np.c_[np.repeat(ar[:, 0], ar.shape[1] - 1),
                               ar[:, 1:].flatten()])
        path.append(np.c_[np.repeat(ar[:, 0], ar.shape[1]),
                          ar[:, :].flatten()])

    return pd.DataFrame(np.concatenate(ancestors),columns=['descendant', 'ancestor']), \
           pd.DataFrame(np.concatenate(path),columns=['descendant', 'path'])


########################################################################################################################
def calculate_distance(positions, decimal=None, type='euclidean'):
    '''
    ex.
    positions=[(0, 0), (3, 4), (7, 7)]
    positions=[(0, 0, 0), (3, 4, 0), (3, 16, 9)]
    '''

    results = []

    # Detect dimension of tuples in the positions
    try:
        if all(len(tup) == 2 for tup in positions):
            dim = 2
        elif all(len(tup) == 3 for tup in positions):
            dim = 3
    except:
        print('Dimension of positions must be same in calculate_distance()!')


    # Calculate distance
    try:
        if all([dim == 2, type == 'haversine']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                lat1 = loc1[0]
                lng1 = loc1[1]

                lat2 = loc2[0]
                lng2 = loc2[1]

                degreesToRadians = (math.pi / 180)
                latrad1 = lat1 * degreesToRadians
                latrad2 = lat2 * degreesToRadians
                dlat = (lat2 - lat1) * degreesToRadians
                dlng = (lng2 - lng1) * degreesToRadians

                a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(latrad1) * \
                math.cos(latrad2) * math.sin(dlng / 2) * math.sin(dlng / 2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                r = 6371000

                results.append(r * c)

        elif all([dim == 2, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]

                x2 = loc2[0]
                y2 = loc2[1]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

                results.append(d)

        elif all([dim == 3, type == 'euclidean']):
            for i in range(1, len(positions)):
                loc1 = positions[i - 1]
                loc2 = positions[i]

                x1 = loc1[0]
                y1 = loc1[1]
                z1 = loc1[2]

                x2 = loc2[0]
                y2 = loc2[1]
                z2 = loc2[2]

                d = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

                results.append(d)


        if decimal is None:
            return sum(results)
        elif decimal == 0:
            return int(round(sum(results)))
        else:
            return round(sum(results), decimal)

    except:
        print('Please use available type and dim, such as "euclidean"(2-dim, 3-dim) and "haversine" (2-dim only), '
              'in calculate_distance().')


########################################################################################################################
def plot_relation_tree(df, child_col, parent_col, label_col=None, view=False, save_path=None,
                       filename='ntree', file_type='pdf', fig_size='6,6', delete_gv_files=True):
    '''
    df = pd.DataFrame({'child': [200, 201, 300, 301, 302, 400],
                   'parent': [100, 100, 200, 200, 201, 300]})

    '''

    u = Digraph(filename, format=file_type)
    u.attr(size=fig_size)
    u.node_attr.update(color='lightblue2', style='filled')

    # Plot the tree
    if label_col is None:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    # View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name



    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)

            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name



    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_relation_tree(df, child_col, parent_col, branch_col=None, polarity_dict=None, node_order_lst=None,
                              show_node_id=True, view=False, save_path=None, filename='ntree', file_type='pdf',
                              fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df[df[parent_col] != -1]

    u = Digraph(filename, format=file_type)
    u.attr(size=fig_size, bgcolor='transparent')


    ### Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_relation_tree().")


    ### Plot nodes
    dict0 = {2: 'green3', 3: 'brown1', 4: 'darkorchid1'}
    # dict0 = {2: 'green4', 3: 'red', 4: 'purple'}
    # dict0 = {2: 'palegreen1', 3: 'lightpink', 4: 'plum'}  # fill in
    if polarity_dict is None:
        u.attr(size=fig_size)
        u.node_attr.update(shape='circle', color='black', style='solid')
    else:
        ### Add type col
        df['type'] = np.where(df[child_col].isin(polarity_dict['axon']), 2, 0)
        df['type'] = np.where(df[child_col].isin(polarity_dict['dendrite']), 3, df['type'])
        df['type'] = np.where(df[child_col].isin(polarity_dict['mix']), 4, df['type'])

        for node in total_node:
            if node != 1:
                t = df.loc[df[child_col]==node, 'type'].values[0]

            if show_node_id:
                node_id = str(node)
            else:
                node_id = ""

            if node == 1:
                u.attr('node', shape='circle', color='black', style='filled')
                u.node(str(node), label=node_id)
            elif t in [2,3,4]:
                u.attr('node', shape='circle', color=dict0[t], style='filled')
                u.node(str(node), label=node_id)
            else:
                u.attr('node', shape='circle', color='black', style='solid')
                u.node(str(node), label=node_id)


    ### Plot the link btw nodes
    if branch_col is None:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child, arrowhead="none")

    else:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(int(round(row[branch_col], 0)))
            u.edge(parent, child, label=label, arrowhead="none")


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)

            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        # u.render("pdf", "pdf", save_path)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_remained_tree(df_dis_origin, df_dis_remained, child_col, parent_col, branch_col=None, polarity_dict=None,
                              node_order_lst=None, show_node_id=True, view=False, save_path=None, filename='rmn_tree',
                              file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis_origin[df_dis_origin[parent_col] != -1]
    df_temp = df_dis_remained[df_dis_remained[parent_col] != -1]

    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_remained_tree().")

    # Create remain nodes
    _, df_path = neuron_ancestors_and_path(df_dis_origin, child_col, parent_col)
    remain_node = []
    for index, row in df_temp.iterrows():
        des_point = row[child_col]
        anc_point = row[parent_col]
        path_points = neuron_path(df_path, des_point, anc_point)
        remain_node = list(set(remain_node) | set(path_points))


    ### Plotting
    # file name & file type
    u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    dict0 = {2: 'green3', 3: 'brown1', 4: 'darkorchid1'}
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in remain_node:
                u.attr('node', shape='circle', color='black', style='dashed')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='circle', color='black', style='solid')
                u.node(str(node))

    else:
        ### Add type col
        df['type'] = np.where(df[child_col].isin(polarity_dict['axon']), 2, 0)
        df['type'] = np.where(df[child_col].isin(polarity_dict['dendrite']), 3, df['type'])
        df['type'] = np.where(df[child_col].isin(polarity_dict['mix']), 4, df['type'])

        for node in total_node:
            if node != 1:
                t = df.loc[df[child_col] == node, 'type'].values[0]

            if show_node_id:
                node_id = str(node)
            else:
                node_id = ""

            if node == 1:
                u.attr('node', shape='circle', color='black', style='filled')
                u.node(str(node), label=node_id)
            # deleted nodes
            elif node not in remain_node:
                if t in [2, 3, 4]:
                    u.attr('node', shape='circle', color=dict0[t], style='dashed')
                    u.node(str(node), label=node_id)
                else:
                    u.attr('node', shape='circle', color='black', style='dashed')
                    u.node(str(node), label=node_id)
            # remained nodes
            else:
                if t in [2, 3, 4]:
                    u.attr('node', shape='circle', color=dict0[t], style='filled')
                    u.node(str(node), label=node_id)
                else:
                    u.attr('node', shape='circle', color='black', style='solid')
                    u.node(str(node), label=node_id)



    # plot the link btw nodes
    if branch_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child, arrowhead="none")

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(int(round(row[branch_col], 0)))
            u.edge(parent, child, label=label, arrowhead="none")


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_reduce_on_level_tree(df_dis_origin, df_dis_remained, child_col, parent_col, label_col=None, polarity_dict=None,
                              node_order_lst=None, view=False, save_path=None, filename='rmn_tree', subname=True,
                              file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis_origin[df_dis_origin[parent_col] != -1]
    df_temp = df_dis_remained[df_dis_remained[parent_col] != -1]

    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_remained_tree().")

    # Create remain nodes
    remain_node = list(set(df_temp["descendant"]) | set(df_temp["ancestor"]))


    ### Plotting
    # file name & file type
    if subname:
        n = int(max(df[child_col]))
        u = Digraph(filename+'_'+str(n), format=file_type)
    else:
        u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in remain_node:
                u.attr('node', shape='ellipse', color='black', style='dashed')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                u.node(str(node))

    else:
        for node in total_node:
            # deleted nodes
            if node not in remain_node:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='doubleoctagon', color='darkorchid1', style='dashed')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='dashed')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='dashed')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='dashed')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='black', style='dashed')
                    u.node(str(node))

            # remained nodes
            else:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='doubleoctagon', color='darkorchid1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))


    # plot the link btw nodes
    if label_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_result_tree(df_dis, df_result, child_col, parent_col, label_col=None, polarity_dict=None,
                            node_order_lst=None, view=False, save_path=None, filename='rslt_tree', subname=True,
                            file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis[df_dis[parent_col] != -1]
    df_temp = df_result

    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_result_tree().")

    # Create inference nodes
    inference_node = df_temp.loc[df_temp['inference']==1, child_col].tolist()
    # _, df_path = neuron_ancestors_and_path(df_dis, child_col, parent_col)
    # remain_node = []
    # for index, row in df_temp.iterrows():
    #     des_point = row[child_col]
    #     anc_point = row[parent_col]
    #     path_points = neuron_path(df_path, des_point, anc_point)
    #     remain_node = list(set(remain_node) | set(path_points))



    ### Plotting
    # file name
    if subname:
        n = int(max(df[child_col]))
        u = Digraph(filename+'_'+str(n), format=file_type)
    else:
        u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in inference_node:
                u.attr('node', shape='ellipse', color='black', style='solid')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                u.node(str(node))

    else:
        for node in total_node:
            # nodes with inference = 0
            if node not in inference_node:
                if node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='solid')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='black', style='solid')
                    u.node(str(node))

            # nodes with inference = 1
            else:
                if node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))


    # plot the link btw nodes
    if label_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_prob_tree(df_dis, df_result, child_col, parent_col, label_col=None, prob_col='prob', polarity_dict=None,
                          node_order_lst=None, prob=[0.5, 0.9], view=False, save_path=None, filename='rslt_tree', subname=True,
                          file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis[df_dis[parent_col] != -1]
    df_temp = df_result

    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_prob_tree().")

    # Create prob nodes
    prob1_node = df_temp.loc[(df_temp[prob_col] > prob[0]) & (df_temp[prob_col] <= prob[1]), child_col].tolist()
    prob2_node = df_temp.loc[df_temp[prob_col] > prob[1], child_col].tolist()



    ### Plotting
    # file name
    if subname:
        n = int(max(df[child_col]))
        u = Digraph(filename+'_'+str(n), format=file_type)
    else:
        u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in any([prob1_node, prob2_node]):
                u.attr('node', shape='ellipse', color='black', style='solid')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                u.node(str(node))

    else:
        for node in total_node:
            # nodes <= prob1 & prob2
            if all([node not in prob1_node, node not in prob2_node]):
                if node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='solid')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='black', style='solid')
                    u.node(str(node))

            # nodes > prob1
            elif node in prob1_node:
                if node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='lightpink', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='palegreen1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='khaki1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))

            # nodes > prob2
            elif node in prob2_node:
                if node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))


    # plot the link btw nodes
    if label_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_mix_prob_tree(df_dis, df_result, child_col, parent_col, label_col=None, axon_prob_col='axon_prob', mix_prob_col='mix_prob', polarity_dict=None,
                              node_order_lst=None, axon_prob=[0.5, 0.9], mix_prob=[0.5, 0.8], view=False, save_path=None, filename='rslt_tree', subname=True,
                              file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis[df_dis[parent_col] != -1]
    df_temp = df_result
    df_temp[axon_prob_col] = np.where(df_temp['mix']==1, df_temp[mix_prob_col], df_temp[axon_prob_col])

    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_mix_prob_tree().")

    # Create prob nodes
    prob1_node = df_temp.loc[(df_temp[axon_prob_col] > axon_prob[0]) & (df_temp[axon_prob_col] <= axon_prob[1]), child_col].tolist()
    prob2_node = df_temp.loc[df_temp[axon_prob_col] > axon_prob[1], child_col].tolist()



    ### Plotting
    # file name
    if subname:
        n = int(max(df[child_col]))
        u = Digraph(filename+'_'+str(n), format=file_type)
    else:
        u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in any([prob1_node, prob2_node]):
                u.attr('node', shape='ellipse', color='black', style='solid')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                u.node(str(node))

    else:
        for node in total_node:
            # nodes <= prob1 & prob2
            if all([node not in prob1_node, node not in prob2_node]):
                if node in polarity_dict['mix']:
                    u.attr('node', shape='ellipse', color='darkorchid1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='solid')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='black', style='solid')
                    u.node(str(node))

            # nodes > prob1
            elif node in prob1_node:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='ellipse', color='plum', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='lightpink', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='palegreen1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='khaki1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))

            # nodes > prob2
            elif node in prob2_node:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='ellipse', color='darkorchid1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))


    # plot the link btw nodes
    if label_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def neuron_plot_realPredMix_tree(df_dis, df_result, child_col, parent_col, label_col=None, axon_prob_col='axon_prob', mix_prob_col='mix_prob', polarity_dict=None,
                                 node_order_lst=None, axon_prob=[0.5, 0.9], mix_prob=[0.5, 0.8], view=False, save_path=None, filename='rslt_tree', subname=True,
                                 file_type='pdf', fig_size='6,6', delete_gv_files=True):

    ### Need to drop row that "PARENT_ID = -1" first
    df = df_dis[df_dis[parent_col] != -1]
    df_temp = df_result.sort_values([mix_prob_col, child_col], ascending=[False, False])  # child_col ascending=F: choose larger ID as mix point


    # real_mix_pt = df_temp.loc[df_temp['mix']==1, child_col].values
    real_mix_pt = polarity_dict["mix"]
    max_prob_mix_pt = np.array(df_temp[child_col])[:len(real_mix_pt)]
    if set(real_mix_pt) != set(max_prob_mix_pt):
        polarity_dict['pred_mix'] = list(max_prob_mix_pt)
        prob1_mix = []
        prob2_mix = list(max_prob_mix_pt)
    else:
        polarity_dict['pred_mix'] = []
        prob1_mix = []
        prob2_mix = list(real_mix_pt)


    # Create total nodes
    if node_order_lst is None:
        total_node = list(set(df[child_col]) | set(df[parent_col]))
    else:
        node_from_df = list(set(df[child_col]) | set(df[parent_col]))
        if set(node_from_df).issubset(node_order_lst):
            total_node = [x for x in node_order_lst if x in node_from_df]
        else:
            sys.exit("\n node_from_df is not a sub-set of node_order_lst! Check neuron_plot_atLeastOne_mix_tree().")

    # Create prob nodes
    # df_m = df_temp.loc[df_temp[child_col].isin([real_mix_pt,  max_prob_mix_pt])]
    # prob1_mix = df_m.loc[(df_m[mix_prob_col] > mix_prob[0]) & (df_m[mix_prob_col] <= mix_prob[1]), child_col].tolist()
    # prob2_mix = df_m.loc[df_m[mix_prob_col] > mix_prob[1], child_col].tolist()

    df_a = df_temp[~df_temp[child_col].isin([real_mix_pt,  max_prob_mix_pt])]
    prob1_axon = df_a.loc[(df_a[axon_prob_col] > axon_prob[0]) & (df_a[axon_prob_col] <= axon_prob[1]), child_col].tolist()
    prob2_axon = df_a.loc[df_a[axon_prob_col] > axon_prob[1], child_col].tolist()

    prob1_node = prob1_mix + prob1_axon
    prob2_node = prob2_mix + prob2_axon



    ### Plotting
    # file name
    if subname:
        n = int(max(df[child_col]))
        u = Digraph(filename+'_'+str(n), format=file_type)
    else:
        u = Digraph(filename, format=file_type)

    u.attr(size=fig_size)

    # plot nodes
    if polarity_dict is None:
        for node in total_node:
            # deleted nodes
            if node not in any([prob1_node, prob2_node]):
                u.attr('node', shape='ellipse', color='black', style='solid')
                u.node(str(node))

            # remained nodes
            else:
                u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                u.node(str(node))

    else:
        for node in total_node:
            # nodes <= prob1 & prob2
            if all([node not in prob1_node, node not in prob2_node]):
                if node in polarity_dict['mix']:
                    u.attr('node', shape='doubleoctagon', color='darkorchid1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['pred_mix']:
                    u.attr('node', shape='diamond', color='darkorchid1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='solid')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='solid')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='black', style='solid')
                    u.node(str(node))

            # nodes > prob1
            elif node in prob1_node:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='doubleoctagon', color='plum', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['pred_mix']:
                    u.attr('node', shape='diamond', color='plum', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='lightpink', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='palegreen1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='khaki1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))

            # nodes > prob2
            elif node in prob2_node:
                if node in polarity_dict['mix']:
                    u.attr('node', shape='doubleoctagon', color='darkorchid1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['pred_mix']:
                    u.attr('node', shape='diamond', color='darkorchid1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['dendrite']:
                    u.attr('node', shape='ellipse', color='brown1', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon']:
                    u.attr('node', shape='ellipse', color='green3', style='filled')
                    u.node(str(node))
                elif node in polarity_dict['axon_drte']:
                    u.attr('node', shape='ellipse', color='gold1', style='filled')
                    u.node(str(node))
                else:
                    u.attr('node', shape='ellipse', color='lightblue2', style='filled')
                    u.node(str(node))


    # plot the link btw nodes
    if label_col is None:
        # don't show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child)

    else:
        # show distance on the link
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(row[label_col])
            u.edge(parent, child, label=label)


    ### View/save the file
    if view:
        # main_folder_name = 'gv_graphs ' + datetime.date.today().strftime('%Y-%m-%d')
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def plot_tree_typePost(df, child_col, parent_col, label_col, pred_col, save_path=None, file_type='pdf', fig_size='6,6', only_terminal=True, branch_col="l", show_node_id=True, view=False, delete_gv_files=True):
    # Create total nodes
    df = df.sort_values(["dp_level", child_col], ascending=[False, True]).reset_index(drop=True)
    total_node = [1] + df.ix[:, child_col].tolist()

    if label_col is not None:
        df[label_col] = np.where(df[label_col] == 0, np.nan, df[label_col])
    if only_terminal:
        df[pred_col] = np.where(df["NC"]!=0, np.nan, df[pred_col])

    # Plotting
    # 1. file name
    filename = df.ix[0, "nrn"]
    if label_col is None:
        filename = "pre_" + filename
    u = Digraph(filename, format=file_type)
    u.attr(size=fig_size, bgcolor='transparent' )
    
    # 2. plot nodes
    dict0 = {2: 'green3', 3: 'brown1', 4: 'darkorchid1', 1:'black'}  # fill in
    # dict0 = {2: 'palegreen1', 3: 'lightpink', 4: 'plum', 1:'black'}  # fill in
    # dict1 = {2: 'green4', 3: 'red', 4: 'purple', 1:'black'}    # shape
    dict1 = {2: 'black', 3: 'black', 4: 'black', 1: 'black'}  # shape
    for node in total_node:
        if show_node_id:
            node_id = str(node)
        else:
            node_id = ""
                        
        # 2.1 assign type to soma
        if node == 1:
            type_true = type_pred = 1
        elif label_col is None:
            type_pred = df.loc[df[child_col] == node, pred_col].values[0]
        else:
            type_true = df.loc[df[child_col]==node, label_col].values[0]
            type_pred = df.loc[df[child_col]==node, pred_col].values[0]

        # 2.2 color the nodes
        if label_col is None:   # WITHOUT label column
            if node == 1:
                u.attr('node', shape='circle', color='black', penwidth='1', style='filled', fillcolor=dict0[type_pred])
                u.node(str(node), label=node_id)
            elif math.isnan(type_pred):
                u.attr('node', shape='circle', color='black', penwidth='1', style='solid')
                u.node(str(node), label=node_id)
            else:
                u.attr('node', shape='circle', color=dict0[type_pred], penwidth='1', style='filled', fillcolor=dict0[type_pred])
                u.node(str(node), label=node_id)
            del type_pred
            
        else:   # label column
            if node == 1:
                u.attr('node', shape='circle', color=dict1[type_true], penwidth='1', style='filled',
                       fillcolor=dict0[type_pred])
                u.node(str(node), label=node_id)
            elif any([math.isnan(type_true), math.isnan(type_pred)]):
                u.attr('node', shape='circle', color='black', penwidth='1', style='solid')
                u.node(str(node), label=node_id)
            elif type_true == type_pred:
                u.attr('node', shape='circle', color=dict0[type_true], penwidth='1', style='filled',
                       fillcolor=dict0[type_pred])
                u.node(str(node), label=node_id)
            else:
                u.attr('node', shape='circle', color=dict1[type_true], penwidth='5', style='filled',
                       fillcolor=dict0[type_pred])
                u.node(str(node), label=node_id)
            del type_true, type_pred

            
    # 3. plot the link btw nodes (show distance on the link)
    if branch_col is None:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            u.edge(parent, child, arrowhead="none")

    else:
        for index, row in df.iterrows():
            child = str(row[child_col])
            parent = str(row[parent_col])
            label = str(int(round(row[branch_col], 0)))
            u.edge(parent, child, label=label, arrowhead="none")


    # View/save the file
    if view:
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.view(directory=Desktop + main_folder_name)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is None]):
        main_folder_name = 'gv_graphs '
        if not os.path.exists(Desktop + main_folder_name):
            try:
                os.makedirs(Desktop + '/' + main_folder_name)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        u.render(directory=Desktop + main_folder_name, view=False)
        path = Desktop + main_folder_name


    elif all([view is False, save_path is not None]):
        u.render(directory=save_path, view=False)
        path = save_path


    if delete_gv_files:
        delete_files_under_directory(directory=path, file_type='gv')

    return


########################################################################################################################
def zero_list_maker(n):
    list_of_zeros = [0] * n
    return list_of_zeros

########################################################################################################################
def neuron_childNumCol(df, child_col='ID', parent_col='PARENT_ID', output_col='NC'):
    ### Create child number col
    df_freq = pd.value_counts(df[parent_col]).to_frame().reset_index()
    df_freq.columns = [child_col, output_col]
    df_freq = df_freq.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_freq, how='left', on=child_col)
    df[output_col] = np.where(np.isnan(df[output_col]), 0, df[output_col])
    df[output_col] = df[output_col].astype(int)

    ### Create list of leaf/fork/root
    tree_node_dict = neuron_tree_node_dict(df, child_col, parent_col, childNum_col=output_col)

    return df, tree_node_dict


########################################################################################################################
def neuron_tree_node_dict(df, child_col='ID', parent_col='PARENT_ID', childNum_col='NC'):
    ### Create list of leaf/fork/root
    leaf_lst = df.loc[df[childNum_col] == 0, child_col].tolist()
    fork_lst = df.loc[df[childNum_col] > 1, child_col].tolist()
    root_lst = df.loc[df[parent_col] == -1, child_col].tolist()
    if len(root_lst) == 1:
        pass
    else:
        sys.exit("\n Multiple roots(somas) in a neuron. Check 'neuron_num_of_child()'.")

    tree_node_dict = {'root': root_lst, 'fork': fork_lst, 'leaf': leaf_lst}

    return tree_node_dict


########################################################################################################################
# def neuron_polarity_dict(df, child_col='ID', type_col='T',
#                          axon=[2, 20, 21, 22, 23], dendrite=[3, 30, 31, 32, 33], axon_drte=[4, 40, 41, 42, 43]):
def neuron_polarity_dict(df, child_col='ID', type_col='T',
                         axon=[2, 20, 21, 22, 23], dendrite=[3, 30, 31, 32, 33], mix_point=[4]):

    axon_lst = df.loc[df[type_col].isin(axon), child_col].tolist()
    den_lst = df.loc[df[type_col].isin(dendrite), child_col].tolist()
    # a_d_lst = df.loc[df[type_col].isin(axon_drte), child_col].tolist()
    mix_lst = df.loc[df[type_col].isin(mix_point), child_col].tolist()

    polarity_dict = {'axon': axon_lst, 'dendrite': den_lst, 'axon_drte': [], 'mix': mix_lst}

    return polarity_dict


########################################################################################################################
def neuron_level_branch(df_path, tree_node_dict):
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']
    ### Create branches (level tuple list)
    level_points = list(set().union(leaf_lst, fork_lst, root_lst))
    df_level = df_path.loc[df_path['path'].isin(level_points)]
    df_level = df_level.loc[df_level['descendant'].isin(leaf_lst)]
    df_level = df_level.reset_index(drop=True)
    # df_level = df_level.sort_values(['descendant', 'path']).reset_index(drop=True)

    branches = []
    for i in df_level.descendant.unique().tolist():
        lst = df_level.loc[df_level['descendant'] == i, 'path'].tolist()
        branches = list(set().union(branches, zip(lst, lst[1:])))

    return branches


########################################################################################################################
def neuron_path(df_path, des_point, anc_point, include_anc_pt=True):
    ### Create path list
    try:
        start = des_point
        end = anc_point

        path_points = df_path.loc[df_path['descendant'] == start, 'path'].tolist()

        start_idx = path_points.index(start)
        end_idx = path_points.index(end)

        if include_anc_pt:
            path_points = path_points[start_idx: (end_idx + 1)]
        else:
            path_points = path_points[start_idx: end_idx]

        return path_points

    except:
        print('No path between the two points. Check neuron_path().')


########################################################################################################################
def neuron_levelCol(df, df_anc, df_path, tree_node_dict, branch_lst, child_col='ID', parent_col='PARENT_ID'):
    ### cihld_col is from "df"
    '''
    df = pd.DataFrame({'ID': [1, 2, 3, 100, 200, 201, 300, 301, 302, 400],
                       'PARENT_ID': [-1, 1, 2, 3, 100, 100, 200, 200, 201, 300]})
    df = pd.DataFrame({'ID': [100, 200, 201, 300, 301, 302, 400],
                       'PARENT_ID': [-1, 100, 100, 200, 200, 201, 300]})
    '''
    leaf_lst = tree_node_dict['leaf']
    fork_lst = tree_node_dict['fork']
    root_lst = tree_node_dict['root']

    ### 1. Points btw root and first_fork
    first_fork = neuron_first_fork(root_lst, fork_lst, branch_lst)
    temp_lst = df_anc.loc[df_anc['descendant'] == first_fork, 'ancestor'].tolist()
    temp_lst.append(first_fork)
    zero_lst = zero_list_maker(len(temp_lst))
    df_temp_1 = pd.DataFrame({child_col: temp_lst, 'level': zero_lst})

    ### 2. Points after first_fork
    df_temp_2 = df_anc.loc[df_anc['ancestor'].isin(fork_lst)]
    df_temp_2 = pd.value_counts(df_temp_2.descendant).to_frame().reset_index()
    df_temp_2.columns = [child_col, 'level']

    ### Merge 1. & 2. to df
    df_temp_1 = pd.concat([df_temp_1, df_temp_2])
    df_temp_1 = df_temp_1.sort_values([child_col]).reset_index(drop=True)

    df = pd.merge(df, df_temp_1, how='left', on=child_col)

    max_level = max(df['level'])


    ### 3. Find deepest level of each point
    df['dp_level'] = 0
    # df_path = df_path.loc[df_path['path'] != 1]

    df_temp = df.loc[df[child_col].isin(leaf_lst), [child_col, 'level']].sort_values(['level'], ascending=False).reset_index(drop=True)
    for l in df_temp[child_col]:
        path = df_path.loc[df_path['descendant'] == l, 'path'].tolist()
        level = df_temp.loc[df_temp[child_col] == l, 'level'].values[0]
        df['dp_level'] = np.where((df[child_col].isin(path)) & (df['dp_level'] < level),
                                  level, df['dp_level'])



    return df, max_level, first_fork


########################################################################################################################
def neuron_first_fork(root_lst, fork_lst, branch_lst):
    soma = root_lst[0]
    if soma in fork_lst:
        first_fork = soma
    else:
        result = [t for t in branch_lst if all([t[1] == soma, t[0] in fork_lst])]
        if len(result) == 1:
            first_fork = result[0][0]
        elif len(result) == 0:
            sys.exit("\n No fork in the neuron! Check 'neuron_first_fork()'.")
        else:
            sys.exit("\n Multiple first_fork in the neuron! Check 'neuron_first_fork()'.")

    return first_fork


########################################################################################################################
def neuron_branchCol_QCol_distance(df, df_path, tree_node_dict, branch_lst, first_fork, decimal=0,
                                   child_col='ID', parent_col='PARENT_ID', type_col='T'):
    ### Create distances of branches and create Q col
    length_lst = []    # distance of branch
    length_lst_soma = []  # distance of descendant to soma
    direct_dis_lst_soma = []  # direct distance of descendant to soma
    df_temp = pd.DataFrame()
    for i in branch_lst:
        start = i[0]
        end = i[1]
        soma = tree_node_dict['root'][0]

        path_points = df_path.loc[df_path['descendant'] == start, 'path'].tolist()

        start_idx = path_points.index(start)
        end_idx = path_points.index(end)
        soma_idx = path_points.index(soma)

        path_points_1 = path_points[start_idx: end_idx]         # exclude the end point(for Q)
        path_points_2 = path_points[start_idx: (end_idx + 1)]   # include the end point(for Q, dis)
        path_points_3 = path_points[start_idx: (soma_idx + 1)]  # path to soma


        # Create branch col and Q col
        # branch with end pt != 1 or first_fork == 1 (first_fork == soma)
        if any([end != 1, first_fork == 1]):
            temp_lst_1 = [start] * len(path_points_1)
            temp_lst_2 = list(range(len(path_points_1)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_1, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)
        # end pt == 1 & first_fork != 1 (first_fork != soma)
        else:
            temp_lst_1 = [start] * len(path_points_2)
            temp_lst_2 = list(range(len(path_points_2)))
            df_temp_1 = pd.DataFrame(OrderedDict({child_col: path_points_2, 'branch': temp_lst_1, 'Q': temp_lst_2}))
            df_temp = df_temp.append(df_temp_1)


        # Calculate distance
        positions = df.loc[df[child_col].isin(path_points_2), ['x', 'y', 'z']]
        tuples = [tuple(x) for x in positions.values]
        positions_s = df.loc[df[child_col].isin(path_points_3), ['x', 'y', 'z']]
        tuples_s = [tuple(x) for x in positions_s.values]
        tuples_ds = [tuples_s[0], tuples_s[-1]]

        length = calculate_distance(tuples, decimal)
        length_lst.append(length)

        length_soma = calculate_distance(tuples_s, decimal)
        length_lst_soma.append(length_soma)

        direct_dis_soma = calculate_distance(tuples_ds, decimal)
        direct_dis_lst_soma.append(direct_dis_soma)


    # Merge branch & Q into original df
    df = pd.merge(df, df_temp, how='left', on=child_col)
    if first_fork == 1:
        df.loc[df[parent_col] == -1, ['branch', 'Q']] = [1, 0]  # add soma if first_fork = 1
    df[['branch', 'Q']] = df[['branch', 'Q']].astype('int')


    ### Create df_dis (cols ['len_des_soma', 'des_T'])
    df_dis = pd.DataFrame({'branch': branch_lst, 'len': length_lst, 'len_des_soma': length_lst_soma, 'direct_dis_des_soma': direct_dis_lst_soma})
    df_dis['ancestor'] = [tuple[1] for tuple in branch_lst]
    df_dis['descendant'] = [tuple[0] for tuple in branch_lst]
    df_dis = df_dis.sort_values(['ancestor', 'descendant']).reset_index(drop=True)
    # # create dis_anc_soma
    # df_anc = df_dis[['descendant', 'len_des_soma']]
    # df_anc = df_anc.rename({'descendant':'ancestor', 'len_des_soma':'dis_anc_soma'})
    # df_dis = pd.merge(df_dis, df_anc, how='left', on='descendant')

    # create type column
    df_t = df[[child_col, type_col]].copy()
    df_t.columns = ['descendant', 'des_T']
    df_dis = pd.merge(df_dis, df_t, how='left', on='descendant')


    ### Reorder columns
    df_dis = df_dis[['branch', 'descendant', 'ancestor', 'len', 'len_des_soma', 'direct_dis_des_soma', 'des_T']]


    # print(df_dis)
    # neuron_plot_relation_tree(df_dis, 'des_point', 'anc_point', 'len', save=True, filename='level')

    return df, df_dis, length_lst


########################################################################################################################
def neuron_LCol(df, df_dis, tree_node_dict, child_col='ID', level_col='level', dpLevel_col='dp_level', branch_col='branch', L_length=None):

    # Parameters
    tree_node_lst = list(set().union(tree_node_dict['leaf'], tree_node_dict['fork'], tree_node_dict['root']))
    level_lst = df[level_col].unique().tolist()
    max_level = max(df[level_col])

    # Prepare df_temp_0
    df_anc = df_dis[['ancestor', 'descendant']]
    df_anc.columns = ['ancestor', child_col]
    df_des = df_dis[['ancestor', 'descendant']]
    df_des.columns = [child_col, 'descendant']

    df_temp_0 = df[[child_col, level_col, dpLevel_col]].copy()
    df_temp_0 = pd.merge(df_temp_0, df_anc, on=child_col, how='left')
    df_temp_0 = pd.merge(df_temp_0, df_des, on=child_col, how='left')
    df_temp_0 = df_temp_0.loc[df_temp_0[child_col].isin(tree_node_lst)]


    ### Create L

    # Create df_L
    if L_length is None:
        df_L = pd.DataFrame(index=range(len(df)), columns=range(max_level))
    elif L_length >= max_level:
        df_L = pd.DataFrame(index=range(len(df)), columns=range(L_length))
    else:
        sys.exit("\n L_length < max_level! Check neuron_LCol().")


    # Loop through each level
    for l in level_lst:
        # Level = 0
        if l == 0:
            # loop through the same level
            df_temp = df_temp_0.loc[df_temp_0[level_col] == l].copy()
            df_temp = df_temp.sort_values([dpLevel_col], ascending=False).reset_index(drop=True)
            same_level = df_temp[child_col].unique().tolist()
            for s in same_level:
                if l == 0:  # Root = [0, 0, ..., 0]
                    temp_i = df.index[df[child_col] == s][0]
                    df_L.ix[temp_i, :] = 0

                    # pass to children
                    des_pt = df_temp.loc[df_temp[child_col] == s, 'descendant'].tolist()
                    for d in des_pt:
                        temp_c = df.index[df[child_col] == d][0]
                        df_L.ix[temp_c, :] = 0

        # Other level (i.e. level with parent)
        else:
            # Group by parent
            anc_pt = df_temp_0.loc[df_temp_0[level_col] == l, 'ancestor'].unique().tolist()
            for a in anc_pt:
                # loop through the same level with same parent
                df_temp = df_temp_0.loc[(df_temp_0['ancestor'] == a) & (df_temp_0[level_col] == l)].copy()
                df_temp = df_temp.sort_values([dpLevel_col], ascending=False).reset_index(drop=True)
                same_level = df_temp[child_col].unique().tolist()
                for s in same_level:
                    temp_i = df.index[df[child_col] == s][0]
                    df_L.ix[temp_i, l - 1] = same_level.index(s) + 1

                    # pass to children
                    des_pt = df_temp.loc[df_temp[child_col] == s, 'descendant'].tolist()
                    for d in des_pt:
                        if np.isnan(d):
                            continue
                        temp_c = df.index[df[child_col] == d][0]
                        df_L.ix[temp_c, :] = df_L.ix[temp_i, :]


    # fill in nan rows
    df_L[[branch_col]] = df[[branch_col]].copy()
    branch_lst = df_L.loc[pd.isnull(df_L[0]), branch_col].unique().tolist()
    for b in branch_lst:
        if b == 0:
            i = df.index[df[child_col] == tree_node_dict['root'][0]][0]
        else:
            i = df.index[df[child_col] == b][0]

        idx = df.index[df[branch_col] == b].tolist()
        for j in idx:
            df_L.ix[j, :-1] = df_L.ix[i, :-1]


    # Create 'L' and 'L_short' columns in df
    del df_L[branch_col]
    df['L'] = df_L.values.tolist()
    df['L_sort'] = df_L[df_L.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(int).astype(str)),axis=1)

    df = df.sort_values(['L_sort', child_col]).reset_index(drop=True)
    order_lst = df[child_col].tolist()


    return df, order_lst


########################################################################################################################
def neuron_scatter_plot(df_dis, leaf_lst, log_y=False, view=False, save_path=None):
    df_dis_total = pd.value_counts(df_dis.dis).to_frame().reset_index()
    df_dis_total.columns = ['len', 'freq']
    df_dis_total['log_dis'] = np.log(df_dis_total['len'])
    df_dis_total['log_freq'] = np.log(df_dis_total['freq'])
    df_dis_total = df_dis_total.sort_values(['len']).reset_index(drop=True)
    # print(df_dis_total)

    df_dis_leaf = df_dis.loc[df_dis['descendant'].isin(leaf_lst)]
    df_dis_leaf = pd.value_counts(df_dis_leaf.dis).to_frame().reset_index()
    df_dis_leaf.columns = ['len', 'freq']
    df_dis_leaf['log_dis'] = np.log(df_dis_leaf['len'])
    df_dis_leaf['log_freq'] = np.log(df_dis_leaf['freq'])
    df_dis_leaf = df_dis_leaf.sort_values(['len']).reset_index(drop=True)
    # print(df_dis_leaf)

    # Check if "leaf_dis_freq <= total_dis_freq"
    df_temp = pd.merge(df_dis_total, df_dis_leaf, on='len', how='left')
    df_temp['freq_x-freq_y'] = df_temp['freq_x'] - df_temp['freq_y']
    less_than_0 = df_temp.loc[df_temp['freq_x-freq_y'] < 0]
    if len(less_than_0) == 0:
        pass
    else:
        sys.exit('\n leaf_dis_freq > total_dis_freq! Check neuron_scatter_plot().')


    if log_y:
        y = 'log_freq'
        y_label = 'Distribution (log)'
    else:
        y = 'freq'
        y_label = 'Distribution'

    sns.regplot(x='len', y=y, data=df_dis_total, fit_reg=False,
                scatter_kws={"s": 80}, order=2, ci=None, truncate=True, label='total')
    sns.regplot(x='len', y=y, data=df_dis_leaf, fit_reg=False,
                scatter_kws={"s": 80}, order=2, ci=None, truncate=True, label='leaf')

    plt.legend()
    plt.title('Total', fontsize=16)
    plt.ylabel(y_label, fontsize=14)
    plt.xlabel('Distance', fontsize=14)

    if view:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


########################################################################################################################
def neuron_hist_plot(df_dis, leaf_lst, kde=False, log_y=False, view=False, save_path=None):
    # 1. histogram
    leaf_dis = df_dis.loc[df_dis['descendant'].isin(leaf_lst), 'len'].values

    if all([kde is True, log_y is True]):
        sys.exit("\n kde and log_y can't both be true in neuron_hist_plot().")
    else:
        pass

    if log_y:
        sns.distplot(np.asarray(df_dis['len']), kde=kde, label='total')
        fig = sns.distplot(leaf_dis, kde=kde, label="leaf")
        fig.set_yscale('log')
        plt.legend()
        plt.title('Total', fontsize=16)
        plt.ylabel('Distribution (log)', fontsize=14)
        plt.xlabel('Distance', fontsize=14)
    else:
        sns.distplot(np.asarray(df_dis['len']), kde=kde, label='total')
        sns.distplot(leaf_dis, kde=kde, label="leaf")
        plt.legend()
        plt.title('Total', fontsize=16)
        plt.ylabel('Distribution', fontsize=14)
        plt.xlabel('Distance', fontsize=14)


    if view:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()


########################################################################################################################
def remove_small_leaf(df, df_dis, tree_node_dict, origin_cols=['x',  'y',  'z',  'R',  'T',  'ID',  'PARENT_ID'],
                      less_than=None, less_than_equal_to=None):
    leaf_lst = tree_node_dict['leaf']

    if less_than is not None:
        small_leaf = df_dis.loc[(df_dis['len'] < less_than) & (df_dis['descendant'].isin(leaf_lst)),
                                'descendant'].tolist()
    elif less_than_equal_to is not None:
        small_leaf = df_dis.loc[(df_dis['len'] <= less_than_equal_to) & (df_dis['descendant'].isin(leaf_lst)),
                                'descendant'].tolist()
    elif all([less_than is not None, less_than_equal_to is not None]):
        small_leaf = df_dis.loc[(df_dis['len'] <= less_than_equal_to) & (df_dis['descendant'].isin(leaf_lst)),
                                'descendant'].tolist()
    else:
        sys.exit('\n No distance threshold in remove_small_leaf().')

    df = df[~df['branch'].isin(small_leaf)].reset_index(drop=True)
    df = df[origin_cols]

    return df


########################################################################################################################
def find_distanceX_of_maxNumY(df_dis, tree_node_dict, dis_depend_on='all', decimal=0, view=False, save_path=None):
    leaf_lst = tree_node_dict['leaf']

    if dis_depend_on == 'all':
        ax = sns.distplot(np.asarray(df_dis['len']), hist_kws={"ec": "k"}, label='total')

    elif dis_depend_on == 'leaf':
        leaf_dis = df_dis.loc[df_dis['descendant'].isin(leaf_lst), 'len'].values
        ax = sns.distplot(leaf_dis, hist_kws={"ec": "k"}, label="leaf")

    else:
        sys.exit("\n Use 'all' or 'leaf' for dis_depend_on parameter in find_distanceX_of_maxNumY().")

    '''
    ex.
    x = np.random.randn(100)
    ax = sns.distplot(x, hist_kws={"ec": "k"})
    data_x, data_y = ax.lines[0].get_data()
    '''
    data_x, data_y = ax.lines[0].get_data()

    max_y = max(data_y)  # Find the maximum y value
    max_x = data_x[data_y.argmax()]  # Find the x value corresponding to the maximum y value
    pylab.text(max_x, max_y, str((max_x, max_y)))
    ax.plot([max_x], [max_y], marker="o")

    # xi = 0 # coordinate where to find the value of kde curve
    # yi = np.interp(xi,data_x, data_y)
    # print ("x={},y={}".format(xi, yi)) # prints x=0,y=0.3698
    # ax.plot([xi],[yi], marker="o")

    if view:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)

    plt.close()

    if decimal is None:
        return max_x
    elif decimal == 0:
        return int(round(max_x))
    else:
        return round(max_x, decimal)


########################################################################################################################
def get_fileNames_from_directory(directory='/Users/csu/Desktop/Neuron/nrn_original', file_type=None, drop_file_type=False):
    file_lst = []
    if directory.endswith("/"):
        directory = directory[:-1]

    for (dirpath, dirnames, filenames) in os.walk(directory):
        file_lst.extend(filenames)
        break

    if all([file_type is not None, file_type != 'all']):
        if type(file_type) is list:
            lst = []
            for t in file_type:
                lst_temp = [x for x in file_lst if x.endswith('.' + t)]
                if drop_file_type:
                    lst_temp = [x.split('.' + file_type)[0] for x in lst_temp]
                lst = list(set(lst)|set(lst_temp))
            file_lst = lst

        elif type(file_type) is str:
            file_lst = [x for x in file_lst if x.endswith('.' + file_type)]
            if drop_file_type:
                file_lst = [x.split('.' + file_type)[0] for x in file_lst]
    # ex. file_type = 'swc' or 'csv' or ['txt', 'csv', 'swc']


    for t in ['.localized', '.DS_Store']:
        if t in file_lst:
            file_lst.remove(t)
        else:
            pass

    file_lst.sort()

    return file_lst


########################################################################################################################
def delete_files_under_directory(directory='/Users/csu/Desktop/Neuron/nrn_plot', file_type='gv'):
    file_lst = []
    if directory.endswith("/"):
        directory = directory[:-1]

    for (dirpath, dirnames, filenames) in os.walk(directory):
        file_lst.extend(filenames)
        break

    if all([file_type is not None, file_type != 'all']):
        if type(file_type) is list:
            lst = []
            for t in file_type:
                lst_temp = [x for x in file_lst if x.endswith('.' + t)]
                lst = list(set(lst)|set(lst_temp))
            file_lst = lst

        elif type(file_type) is str:
            file_lst = [x for x in file_lst if x.endswith('.' + file_type)]
    # ex. file_type = 'swc' or 'csv' or ['txt', 'csv', 'swc']


    for file in file_lst:
        os.remove(directory + '/' + file)

    return


########################################################################################################################
def delete_folder_under_directory(directory='/Users/csu/Desktop/123'):
    shutil.rmtree(directory)
    return


########################################################################################################################
def check_L_form(mylist, idx=0):
    # Check if the list fits L form,
    # i.e. elements before the idx are non-zero &
    # elements after (including) the idx are zero
    b = mylist[:idx]
    a = mylist[idx:]
    before = all(p != 0 for p in b)
    after = all(p == 0 for p in a)

    if all([before, after]):
        return True
    else:
        return False


########################################################################################################################
def create_total_L(level=3, branch=2):
    '''
    ### Calculate the total number of L
    a = 1
    r = branch
    length = level + 1
    geometric_series = [a * r ** (n - 1) for n in range(1, length + 1)]
    n = sum(geometric_series)
    '''


    ### Create total L
    if isinstance(branch, int):
        l1 = [list(range(1, branch+1))]     # level 0~1
        l2 = [list(range(branch+1)) for x in range(1, level)]   # level 1~2, 2~3,...
        l = l1+l2
    elif isinstance(branch, list):
        l = None
        for idx, val in enumerate(branch):
            if idx == 0:    # level 0~1
                l1 = [list(range(1, val+1))]
                l = l1
            else:           # level 1~2, 2~3,...
                l2 = [list(range(val+1))]
                l += l2


    # level 0
    L0 = [zero_list_maker(level)]
    # level 1, 2,...
    L1 = list(it.product(*l))       # create combinations
    L1 = [list(t) for t in L1]      # turn tuples into lists

    L1 = L0 + L1



    df_temp = pd.DataFrame(L1)
    m = df_temp.mask(df_temp == 0)
    fs = [pd.Series.first_valid_index, pd.Series.last_valid_index]
    r0 = pd.concat([m.apply(f, 1) for f in fs], axis=1, keys=['first', 'last'])

    m = df_temp.mask(df_temp != 0)
    fs = [pd.Series.first_valid_index]
    r1 = pd.concat([m.apply(f, 1) for f in fs], axis=1, keys=['zero'])

    df_temp = pd.concat([df_temp, r0, r1], axis=1)


    df_temp["keep"] = 'F'
    df_temp["keep"] = np.where((np.isnan(df_temp["first"])) & (np.isnan(df_temp["last"])), 'T',df_temp["keep"])
    df_temp["keep"] = np.where((np.isnan(df_temp["zero"])) | (df_temp["first"]==df_temp["last"]), 'T', df_temp["keep"])
    df_temp["keep"] = np.where((df_temp["zero"] > df_temp["first"]) & (df_temp["zero"] > df_temp["last"]), 'T', df_temp["keep"])
    df_temp = df_temp[df_temp['keep'] == 'T']
    df_temp = df_temp[list(range(level))]
    df = df_temp.copy()



    # Create 'L' and 'L_short' columns in df
    # del df_L[branch_col]
    df['L'] = df_temp.values.tolist()
    df['L_sort'] = df_temp[df_temp.columns[0:]].apply(lambda x: ''.join(x.dropna().astype(int).astype(str)), axis=1)
    df = df.sort_values(['L_sort']).reset_index(drop=True)

    L = df["L"].tolist()
    L_sort = df['L_sort'].tolist()
    n = len(df)


    return L, L_sort, n


########################################################################################################################
def list_unique(mylist):
    x = np.array(mylist)
    x = list(np.unique(x))
    x.sort()
    return x


########################################################################################################################
def merge_multiple_df(df_lst, id_col_lst, param_lst=None, normalize_lst=None):
    '''
    df_lst=[df_main, df1, df2,..., dfn]
    id_col_lst=['id_main', 'id1', 'id2',..., idn]
    '''
    df_0 = df_lst[0]        # df_main
    id_0 = id_col_lst[0]    # id_main

    temp_lst = df_0[id_0].unique()
    df_result = pd.DataFrame({id_0: temp_lst})

    # Change every df's id into id_0
    for idx, val in enumerate(df_lst):
        id_n = id_col_lst[idx]
        val = val.rename(columns={id_n: id_0})
        df_lst[idx] = val

    # Check if param_lst must not be empty or none
    if any([param_lst is None, not param_lst]):
        sys.exit("\n param_lst must not be empty! Check prepare_df().")

    # Add params to df_result
    if normalize_lst is None:
        for i in param_lst:
            df_temp = None
            for df in df_lst:
                if i in df.columns:
                    df_temp = df
                    df_temp = df_temp[[id_0, i]]
                    df_temp = df_temp.drop_duplicates(subset=[id_0, i], keep="first")
                    break

            if df_temp is None:
                sys.exit("\n Can't find items of param_lst in any df! Check prepare_df().")

            df_result = pd.merge(df_result, df_temp, how='left', on=id_0)

    else:
        for i in param_lst:
            df_temp = None
            for df in df_lst:
                if i in df.columns:
                    df_temp = df
                    df_temp = df_temp[[id_0, i]]
                    df_temp = df_temp.drop_duplicates(subset=[id_0, i], keep="first")
                    break

            if df_temp is None:
                sys.exit("\n Can't find items of param_lst in any df! Check prepare_df().")

            if i in normalize_lst:
                max_i = max(df_temp[i])
                df_temp[i] = df[i]/max_i
                df_temp = df_temp.rename(columns={i: "norm_"+i})

            df_result = pd.merge(df_result, df_temp, how='left', on=id_0)



    return df_result


########################################################################################################################
def expand_df_dis(df_dis, ignore_first_fork=None, avg_dis_des=True):
    id_lst = list(set().union(df_dis['ancestor'], df_dis['descendant']))
    df = pd.DataFrame({'ID':id_lst})
    df_temp = df_dis[['descendant', 'len_des_soma']]
    df_temp = df_temp.rename(columns={'descendant': 'ID', 'len_des_soma': 'dis_soma'})
    df = pd.merge(df, df_temp, how='left', on='ID')
    df['dis_soma'] = np.where(df['ID'] == 1, 0, df['dis_soma'])
    max_d = max(df['dis_soma'])
    df['norm_dis_soma'] = df['dis_soma']/max_d

    if isinstance(ignore_first_fork, (int, np.integer)):
        df = df[df['ID'] != ignore_first_fork].reset_index(drop=True)
        df_r = df_dis[['ancestor', 'descendant']]
        df_r = df_r[df_r['descendant'] != ignore_first_fork].reset_index(drop=True)
        df_r['ancestor'] = np.where(df_r['ancestor'] == ignore_first_fork, 1, df_r['ancestor'])
    elif ignore_first_fork is not None:
        sys.exit("\n 'ignore_first_fork' must be int or None! Check expand_df_dis().")

    df_temp = df.copy()

    df_a = df_r
    df_a = df_a.rename(columns={'descendant':'ID'})
    df = pd.merge(df, df_a, how='left', on='ID')
    df_temp_a = df_temp.rename(columns={'ID':'ancestor', 'dis_soma':'dis_anc_soma', 'norm_dis_soma':'norm_dis_anc_soma'})
    df = pd.merge(df, df_temp_a, how='left', on='ancestor')
    df['ancestor'] = np.where(df['ID'] == 1, -1, df['ancestor'])
    df['dis_anc_soma'] = np.where(df['ID'] == 1, 0, df['dis_anc_soma'])
    df['norm_dis_anc_soma'] = np.where(df['ID'] == 1, 0, df['norm_dis_anc_soma'])
    df[['ancestor']] = df[['ancestor']].astype('int')

    df_b = df_r
    df_b = df_b.rename(columns={'ancestor': 'ID'})
    df = pd.merge(df, df_b, how='left', on='ID')
    df_temp_b = df_temp.rename(columns={'ID':'descendant', 'dis_soma':'len_des_soma', 'norm_dis_soma':'norm_len_des_soma'})
    df = pd.merge(df, df_temp_b, how='left', on='descendant')
    df['len_des_soma'] = np.where(pd.isnull(df['descendant']), df['dis_soma'], df['len_des_soma'])
    df['norm_len_des_soma'] = np.where(pd.isnull(df['descendant']), df['norm_dis_soma'], df['norm_len_des_soma'])

    if avg_dis_des:
        id_lst = df['ID'].unique().tolist()
        for i in id_lst:
            d = np.mean(df.loc[df['ID']==i, 'len_des_soma'].values)
            nd = np.mean(df.loc[df['ID']==i, 'norm_len_des_soma'].values)
            df['descendant'] = np.where(df['ID']==i, 'avg', df['descendant'])
            df['len_des_soma'] = np.where(df['ID']==i, d, df['len_des_soma'])
            df['norm_len_des_soma'] = np.where(df['ID']==i, nd,df['norm_len_des_soma'])

        df = df.drop_duplicates(subset=['ID', 'len_des_soma', 'norm_len_des_soma'], keep="first").reset_index(drop=True)

    return df


########################################################################################################################
def find_mix_point(df, df_dis, tree_node_dict, polarity_dict, child_col="ID", parent_col="PARENT_ID"):
    # 1. Use df_dis to find mix_point
    _a_lst = df_dis.ancestor.unique()
    _df_dis = df_dis.sort_values(["len"], ascending=[False])

    mp_lst = []

    for a in _a_lst:
        _t_lst = _df_dis.loc[_df_dis["ancestor"] == a, "des_T"].tolist()

        if len(_t_lst) == 1:
            continue
        elif len(_t_lst) > 2:
            _t_lst = _t_lst[:2]

        if any([_t_lst[0]==0, _t_lst[1]==0, _t_lst[0] == _t_lst[1]]):
            pass
        else:
            mp_lst.append(a)

    # 2. If 1. fail, them use bottom up method to find mix_point
    if not mp_lst:
        leaf_lst = tree_node_dict['leaf']
        axon_lst = polarity_dict['axon']
        dend_lst = polarity_dict['dendrite']

        a_leaf = list(set(axon_lst) & set(leaf_lst))
        d_leaf = list(set(dend_lst) & set(leaf_lst))

        # Find the path of every point
        _, df_path = neuron_ancestors_and_path(df, child_col, parent_col)

        # axon meet dendrite
        dp_lst = []
        for d in d_leaf:
            path_points = neuron_path(df_path, d, 1, include_anc_pt=True)
            dp_lst += path_points
        dp_lst = list_unique(dp_lst)

        mp_lst0 = []
        for a in a_leaf:
            path_points = neuron_path(df_path, a, 1, include_anc_pt=True)
            for p in path_points:
                if p in dp_lst:
                    mp_lst0.append(p)
                    break

        '''
        # dendrite meet axon
        ap_lst = []
        for a in a_leaf:
            path_points = neuron_path(df_path, a, 1, include_anc_pt=True)
            ap_lst += path_points
        ap_lst = list_unique(ap_lst)

        mp_lst1 = []
        for d in d_leaf:
            path_points = neuron_path(df_path, d, 1, include_anc_pt=True)
            for p in path_points:
                if p in ap_lst:
                    mp_lst1.append(p)
                    break

        mp_lst = list(set(mp_lst0) & set(mp_lst1))
        mp_lst = list_unique(mp_lst)
        '''

        mp_lst = list_unique(mp_lst0)



    return mp_lst


########################################################################################################################
'''
def find_mix_point(df, tree_node_dict, polarity_dict, child_col="ID", parent_col="PARENT_ID"):
    leaf_lst = tree_node_dict['leaf']
    axon_lst = polarity_dict['axon']
    dend_lst = polarity_dict['dendrite']

    a_leaf = list(set(axon_lst) & set(leaf_lst))
    d_leaf = list(set(dend_lst) & set(leaf_lst))

    # Find the path of every point
    _, df_path = neuron_ancestors_and_path(df, child_col, parent_col)

    # axon meet dendrite
    dp_lst = []
    for d in d_leaf:
        path_points = neuron_path(df_path, d, 1, include_anc_pt=True)
        dp_lst += path_points
    dp_lst = list_unique(dp_lst)

    mp_lst0 = []
    for a in a_leaf:
        path_points = neuron_path(df_path, a, 1, include_anc_pt=True)
        for p in path_points:
            if p in dp_lst:
                mp_lst0.append(p)
                break

    
    # # dendrite meet axon
    # ap_lst = []
    # for a in a_leaf:
    #     path_points = neuron_path(df_path, a, 1, include_anc_pt=True)
    #     ap_lst += path_points
    # ap_lst = list_unique(ap_lst)
    # 
    # mp_lst1 = []
    # for d in d_leaf:
    #     path_points = neuron_path(df_path, d, 1, include_anc_pt=True)
    #     for p in path_points:
    #         if p in ap_lst:
    #             mp_lst1.append(p)
    #             break
    # 
    # mp_lst = list(set(mp_lst0) & set(mp_lst1))
    # mp_lst = list_unique(mp_lst)
    

    mp_lst = list_unique(mp_lst0)



    return mp_lst
'''

########################################################################################################################

def find_firstFork_by_LQCol(df, child_col, L_sort_col='L_sort', Q_col='Q'):
    L_0 = min(df[L_sort_col])
    first_fork = df.loc[(df[L_sort_col] == L_0) & (df[Q_col] == 0), child_col].values[0]

    return first_fork


########################################################################################################################
def expand_level_of_L(df, level=5, L_col='L', L_sort_col='L_sort'):
    l_0 = len(df.ix[0, L_col])
    if l_0 >= level:
        pass
    # elif l_0 < level:
    else:
        l_1 = level - l_0
        for index, row in df.iterrows():
            row[L_col] += zero_list_maker(l_1)
        df[L_sort_col] = df[L_col].apply(lambda x: ''.join(map(str, x)))
        df = df.sort_values(['L_sort']).reset_index(drop=True)
    # else:
    #     sys.exit("\n level must bigger than origin L! Check expand_level_of_L().")

    return df


########################################################################################################################
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    axes = plt.gca()
    axes.set_ylim([0, 1])

    plt.legend(loc="best")
    return plt


########################################################################################################################
def classification(train_df, test_df, label, features, model, standardization=False,
                   pred_threshold=None, cv=False):  # Info of classifiers.
    # y
    # check if there is a label col in test_df
    for col in label:
        if col in test_df.columns:
            y_test = np.array(test_df[label[0]])
        else:
            y_test = None
    y_train = np.array(train_df[label[0]])

    # x
    X_test = np.array(test_df[features])
    X_train = np.array(train_df[features])

    ### Run (w/ standardize)
    if standardization:
        scaler = preprocessing.StandardScaler().fit(X_train)
        model.fit(scaler.transform(X_train), y_train)

        ### Predict output: 1/0, threshold: default
        if pred_threshold is None:
            pred_proba = model.predict_proba(scaler.transform(X_test))
            y_pred = model.predict(scaler.transform(X_test))
            # y_proba = pred_proba[:, 1]

        ### Predict output: 1/0, threshold: float
        else:
            try:
                pred_proba = model.predict_proba(scaler.transform(X_test))
                y_pred = np.where(pred_proba[:, 1] >= pred_threshold, 1, 0)

            except:
                print('Invalid settings!')




                ## Simon
                # scaler = preprocessing.StandardScaler().fit(X_train.reshape(len(X_train), -1))
                # # scaler1 = preprocessing.StandardScaler().fit(X_train)
                # # a = (scaler.mean_ == scaler1.mean_)
                # # b = (scaler.std_ == scaler1.std_)
                # model.fit(scaler.transform(X_train.reshape(len(X_train), -1)), y_train)
                # y_pred = model.predict(scaler.transform(X_test.reshape(len(X_test), -1)))

    ### Run (w/o standardize)
    else:
        model.fit(X_train, y_train)

        ### Predict output: 1/0, threshold: default
        if pred_threshold is None:
            pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

        ### Predict output: 1/0, threshold: float
        else:
            try:
                pred_proba = model.predict_proba(X_test)
                y_pred = np.where(pred_proba[:, 1] >= pred_threshold, 1, 0)

            except:
                print('Invalid pred_threshold!')





                ## Simon
                # model.fit(X_train.reshape(len(X_train), -1), y_train)
                # y_pred = model.predict(X_test.reshape(len(X_test), -1))



    if cv is True:
        title = "Learning Curves "
        # SVC is more expensive so we do a lower number of CV iterations:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        estimator = model
        plot_learning_curve(estimator, title, X_train, y_train, (0.7, 1.01), cv=cv, n_jobs=4)
    else:
        pass




    return y_test, y_pred, pred_proba, model

########################################################################################################################
def evaluate_regressor_as_binary(actual_y, pred_y):
    # matrix = pd.crosstab(pred_y, actual_y, rownames = ['Predict'], colnames = ['Truth'])
    # true_positive = matrix.values[1, 1] / len(actual_y)
    test_precision = precision_score(actual_y, pred_y)
    test_accuracy = accuracy_score(actual_y, pred_y)
    test_recall = recall_score(actual_y, pred_y)
    test_f1 = f1_score(actual_y, pred_y)

    # result = {'true_positive': true_positive, 'test_precision': test_precision, 'test_accuracy': test_accuracy, 'test_recall': test_recall,
    #           'test_f1': test_f1}

    result = {'test_precision': test_precision, 'test_accuracy': test_accuracy, 'test_recall': test_recall, 'test_f1': test_f1}

    for k, v in result.items():
            result[k] = v*100
    #         evaluate_regressor = json.dumps(result)
    #
    #
    # return evaluate_regressor
    return result


########################################################################################################################
def calculate_nrn_max_length(df, tree_node_dict, child_col, parent_col, decimal=0):
    _, df_path = neuron_ancestors_and_path(df, child_col, parent_col)

    length_lst= []
    for i in tree_node_dict['leaf']:
        # Calculate distance
        root = tree_node_dict['root'][0]
        path_points = neuron_path(df_path, i, root, include_anc_pt=True)
        positions = df.loc[df[child_col].isin(path_points), ['x', 'y', 'z']]
        tuples = [tuple(x) for x in positions.values]

        len = calculate_distance(tuples, decimal)
        length_lst.append(len)

    length = max(length_lst)

    return length


########################################################################################################################
def assign_root_as_firstFork(df_dis, tree_node_dict, child_col='descendant', parent_col='ancestor'):
    root = tree_node_dict['root'][0]
    freq = len(df_dis.loc[df_dis[parent_col] == root])
    if freq == 1:
        orig_first_fork = df_dis.loc[df_dis[parent_col] == root, child_col].values[0]
        df_dis['len'] = np.where(df_dis[parent_col] == orig_first_fork, df_dis['len_des_soma'], df_dis['len'])
        df_dis[parent_col] = np.where(df_dis[parent_col] == orig_first_fork, root, df_dis[parent_col])
        df_dis = df_dis[df_dis[child_col] != orig_first_fork]

    else:
        pass

    return df_dis, orig_first_fork


########################################################################################################################
def find_L_descendant(anc_L, L_lst, include_anc_L=True, turn_to_str=True):
    L_des_lst = []

    if 0 not in anc_L:
        L_des_lst.append(anc_L)
    else:
        idx0 = anc_L.index(0)
        trg_L = anc_L[:idx0:]
        for l in L_lst:
            l_temp = l[:idx0:]
            if l_temp == trg_L:
                L_des_lst.append(l)
            else:
                continue

    if include_anc_L is False:
        L_des_lst = [x for x in L_des_lst if x != anc_L]

    if turn_to_str:
        L_des_lst = [''.join(map(str, a)) for a in L_des_lst]

    return L_des_lst


########################################################################################################################
def detect_branch_of_each_level(df, L_col='L'):
    branch = []
    level = len(df[L_col][0])
    lst = df[L_col].tolist()
    df_temp = pd.DataFrame(lst)

    for i in range(0, level):
        branch.append(max(df_temp[i]))

    return branch


########################################################################################################################
def new_L_for_sub_nrn(df, child_col, anc_pt, L_col='L', L_sort_col='L_sort'):
    anc_L = df.loc[df[child_col]==anc_pt, L_col].values[0]
    level = len(anc_L)
    if 0 not in anc_L:
        idx0 = len(anc_L)
    else:
        idx0 = anc_L.index(0)

    for idx, row in df.iterrows():
        val_L = row[L_col][idx0::] + zero_list_maker(idx0)
        val_L_sort = ''.join(map(str, val_L))
        df.at[idx, L_col] = val_L
        df.at[idx, L_sort_col] = val_L_sort
    # df[L_sort_col] = df[L_col].apply(lambda x: ''.join(map(str, x)))  # Can't use this because df[L_short_col] is already exist!
    df = df.sort_values(['L_sort']).reset_index(drop=True)

    df = expand_level_of_L(df, level, L_col, L_sort_col)

    return df


########################################################################################################################
def new_L_for_sub_bush(df, child_col, anc_pt, feature_level=5, L_col='L', L_sort_col='L_sort'):
    _level = df.loc[df[child_col] == anc_pt, 'level'].values[0] + feature_level
    _des_lst = df.loc[df[child_col] == anc_pt, 'bush'].values[0]

    # if anc_pt is leaf
    if type(_des_lst)!=list:
        df = df.loc[df[child_col] == anc_pt].reset_index(drop=True)
        df[L_col] = [zero_list_maker(feature_level)]
        df[L_sort_col] = df[L_col].apply(lambda x: ''.join(map(str, x)))

    else:
        _lst = [anc_pt] + _des_lst
        df = df.loc[(df['level'] <= _level) & (df[child_col].isin(_lst))].reset_index(drop=True)

        # Find the number of nonzero (x) and the number of zero (y)
        anc_L = df.loc[df[child_col] == anc_pt, L_col].values[0]
        l = len(anc_L)
        if 0 not in anc_L:
            x = l
            y = 0
        else:
            x = anc_L.index(0)
            y = l - x

        # Recreate the L and L_shoort
        if y == feature_level:
            for idx, row in df.iterrows():
                val_L = row[L_col][x::]
                val_L_sort = ''.join(map(str, val_L))
                df.at[idx, L_col] = val_L
                df.at[idx, L_sort_col] = val_L_sort

        elif y < feature_level:
            for idx, row in df.iterrows():
                val_L = row[L_col][x::] + zero_list_maker(feature_level-y)
                val_L_sort = ''.join(map(str, val_L))
                df.at[idx, L_col] = val_L
                df.at[idx, L_sort_col] = val_L_sort
        else:
            for idx, row in df.iterrows():
                val_L = row[L_col][x:x+feature_level:]
                val_L_sort = ''.join(map(str, val_L))
                df.at[idx, L_col] = val_L
                df.at[idx, L_sort_col] = val_L_sort

        # df[L_sort_col] = df[L_col].apply(lambda x: ''.join(map(str, x)))  # Can't use this because df[L_short_col] is already exist!
        df = df.sort_values(['L_sort']).reset_index(drop=True)

    return df


########################################################################################################################
def partition(lst, n=None, pct=None, shuffle_list=True):
    if shuffle_list:
        random.shuffle(lst)

    if all([n is not None, pct is None]):
        division = len(lst) / n
    elif all([n is None, pct > 0, pct < 1]):
        val = 1/pct
        # if (float(val) % 1) >= 0.5:
        #     n = math.ceil(val)
        # else:
        #     n = round(val)
        n = round(val)
        division = len(lst)/n
    else:
        sys.exit("\n Use either Number(n=1, 2, 3...) or Percent(pct=0.1, 0.2, 0.3,...) to separate the list.")

    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

'''
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]
'''

########################################################################################################################
def list_sampling(lst, n=None, pct=None, with_replacement=False, sampling_times=10, shuffle_list=True, cover_all=True):
    '''
    :param lst: target list
    :param n: separate lst into n groups
    :param pct: separate lst into (1/pct) groups with (len(lst)*pct) in every group
    pct=0.672 -> test_set=154, train+cv=60+15
    pct=0.562 -> test_set=129, train+cv=80+20
    pct=0.452 -> test_set=104, train+cv=100+25
    pct=0.345 -> test_set=79, train+cv=120+30
    pct=0.126 -> test_set=29, train+cv=160+40
    '''

    if shuffle_list:
        random.shuffle(lst)

    sample_set = []
    remain_set = []
    if with_replacement:
        if all([n is None, pct > 0, pct < 1]):
            num = round(pct*len(lst))

            if cover_all:
                # make sure every element has been selected once
                if pct*sampling_times < 1:
                    sys.exit("\n 'sampling_times' is too small to 'cover_all_data'. Check list_sampling().")

                remain_times = sampling_times
                while True:
                    # evenly divide the target list into lists
                    random.shuffle(lst)
                    _sampling_lst = [lst[i:i + num] for i in range(0, len(lst), num)]
                    _l = _sampling_lst[-2] + _sampling_lst[-1]
                    _l = _l[-num:]
                    _sampling_lst = _sampling_lst[:-1]
                    _sampling_lst.append(_l)

                    # update sampling_lst and remain_time
                    if remain_times >= len(_sampling_lst):
                        remain_times -= len(_sampling_lst)
                        sample_set += _sampling_lst
                        if remain_times == 0:
                            break
                    else:
                        _sampling_lst = _sampling_lst[:remain_times]
                        sample_set += _sampling_lst
                        break


            else:
                sample_set = []
                for i in range(sampling_times):
                    x = random.sample(lst, num)
                    sample_set.append(x)

        else:
            sys.exit("\n 'with_replacement' and 'sampling_times' can be used only with 'Percent'(pct=0.1, 0.2, 0.3,...). Check list_sampling().")



    else:
        if all([n is not None, pct is None]):
            if n > len(lst):
                sys.exit("\n n > len(lst)! Check list_sampling().")
            else:
                division = len(lst) / n
        elif all([n is None, pct > 0, pct < 1]):
            val = 1 / pct
            n = round(val)
            division = len(lst) / n
        else:
            sys.exit("\n Use either Number(n=1, 2, 3...) or Percent(pct=0.1, 0.2, 0.3,...) to separate the list. Check list_sampling().")

        sample_set = [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

    # remain_set
    for i in range(len(sample_set)):
        _lst = list(set(lst) - set(sample_set[i]))
        remain_set.append(_lst)

    return sample_set, remain_set


########################################################################################################################
def dict_sampling(d, pct=None, num=None, sample_times=10, min=1):
    '''
    pct=0.87 -> sample_set=160+40
    pct=0.655 -> sample_set=120+30
    pct=0.54 -> sample_set=100+25
    pct=0.43 -> sample_set=80+20
    pct=0.325 -> sample_set=60+15
    '''
    sample_set = []
    remain_set = []
    total_lst = dict_merge_value(d)
    for i in range(sample_times):
        _lst = []
        for k, v in d.items():
            if len(v) <= min:
                _lst += v
            else:
                try:
                    if all([type(num) is int, pct is None]):
                        n = num
                    elif all([num is None, 0<pct<1]):
                        n = round(len(v)*pct)
                except:
                    sys.exit("\n Use num(int) or pct(0,1) correctly! Check dict_sampling().")

                random.shuffle(v)
                v = v[:n]
                _lst += v

        _r_lst = list(set(total_lst)-set(_lst))
        sample_set.append(_lst)
        remain_set.append(_r_lst)

    return sample_set, remain_set

########################################################################################################################
def list_freq(lst):
    d = {}
    for i in lst:
        if d.get(i):
            d[i] += 1
        else:
            d[i] = 1
    return d


########################################################################################################################
def dict_remove_key(d, key):
    r = dict(d)
    del r[key]
    return r


########################################################################################################################
def dict_merge_value(d, unique=True):
    lst = []
    for k, v in d.items():
        if lst is None:
            lst = v
        else:
            lst += v

    if unique:
        lst = list_unique(lst)

    return lst


########################################################################################################################
def get_nrn_dir(nrn_name, remove_method=None, target_level=None, input_folder='./data/'):
    if all([remove_method is None, target_level is None]):
        nrn_dir = input_folder + nrn_name + ".pkl"
        return nrn_dir
    else:
        nrn_dir = input_folder + '_'.join([nrn_name.split(".swc")[0], remove_method, target_level])+".pkl"
        return nrn_dir
########################################################################################################################
def num_digits(n):
    """
    >>> num_digits(12345)
    5
    >>> num_digits(0)
    1
    >>> num_digits(-12345)
    5
    """
    if n == 0:
        return 1
    count = 0
    while abs(n) >= 1:
        count += 1
        n = abs(n) / 10
    return count
########################################################################################################################
def axmx_feature_dict(target_level, branch):
    '''
    Prepare axon and mix feature dict
    _, L_sort_lst, _ = create_total_L(target_level, branch)
    d = ["d_" + i for i in L_sort_lst]
    pr = ["pr_" + i for i in L_sort_lst]
    feature_dict = {"f1": ['s_0'] + d,
                    "f2": ['s_0'] + d + pr,
                    "f3": ['s_0'] + d + ['npr_0', 'npr_1'],
                    "f4": ['s_0'] + d + ['ppr_0', 'ppr_1'],
                    "f5": pr,
                    "f6": ['npr_0', 'npr_1'],
                    "f7": ['ppr_0', 'ppr_1']}
    '''
    _, L_sort_lst, _ = create_total_L(target_level, branch)
    s = ["norm_s_" + i for i in L_sort_lst]
    l = ["norm_len_" + i for i in L_sort_lst]
    ds = ["norm_direct_s_" + i for i in L_sort_lst]
    pr = ["pr_" + i for i in L_sort_lst]
    feature_dict = {"f1": s,
                    "f2": l,
                    "f3": ds,
                    "f4": s + l + ds,
                    "f5": s + l,
                    "f6": s + ds,
                    "f7": l + ds}

    return feature_dict


########################################################################################################################
def create_axon_feature_dict(target_level, branch):
    _, L_sort_lst, _ = create_total_L(target_level, branch)

    # distance to parent
    l = ["l_" + i for i in L_sort_lst]
    nl = ["nl_" + i for i in L_sort_lst]

    # distance to soma
    s0 = ["s_" + L_sort_lst[0]]  # use first node only
    s = ["s_" + i for i in L_sort_lst]  # length
    ns = ["ns_" + i for i in L_sort_lst]
    ds = ["ds_" + i for i in L_sort_lst]  # direct distance
    nds = ["nds_" + i for i in L_sort_lst]

    # shape
    ro = ["ro_" + i for i in L_sort_lst]
    c = ["c_" + i for i in L_sort_lst]

    # specific select
    # csu todo: need normalized ratio of children
    f0 = ["ds_", "s_", "nds_", "ns_", "nl_", "ro_", "c_", "rc_"]
    f0 = [i + L_sort_lst[0] for i in f0]




    # feature combinations
    feature_dict = {"af1": ns,
                    "af2": nl,
                    "af3": nds,
                    "af4": ns + nl,
                    "af5": ns + nds,
                    "af6": nl + nds,
                    "af7": ns + nl + nds,
                    "af8": s + ns + ds + nds + l + nl,
                    "af9": s + ns + ds + nds,
                    "af10": s + ns + ds + nds + l + nl + ro + c,
                    "af11": f0

                    }

    return feature_dict


########################################################################################################################
def create_mix_feature_dict(target_level, branch, model):
    _, L_sort_lst, _ = create_total_L(target_level, branch)

    # distance to parent
    l = ["len_" + i for i in L_sort_lst]
    n_l = ["norm_len_" + i for i in L_sort_lst]

    # distance to soma
    s0 = ["s_" + L_sort_lst[0]]     # use first node only
    s = ["s_" + i for i in L_sort_lst]  # length
    n_s = ["norm_s_" + i for i in L_sort_lst]
    ds = ["direct_s_" + i for i in L_sort_lst]   # direct distance
    n_ds = ["norm_direct_s_" + i for i in L_sort_lst]

    # axon pred prob
    pr = [model + "_" + i for i in L_sort_lst]
    npr = [model + "_npr_0", model + "_npr_1"]
    ppr = [model + "_ppr_0", model + "_ppr_1"]

    # normalized difference of terminal
    n_dtm = ["norm_diff_trm_" + i for i in L_sort_lst]

    # feature combinations
    feature_dict = {"mf1": n_s,
                    "mf2": n_l,
                    "mf3": n_ds,
                    "mf4": n_s + n_l,
                    "mf5": n_s + n_ds,
                    "mf6": n_l + n_ds,
                    "mf7": n_s + n_l + n_ds,
                    "mf8": n_s + n_l + pr,
                    "mf9": n_s + n_l + npr,
                    "mf10": n_s + n_l + ppr,
                    "mf11": pr,
                    "mf12": npr,
                    "mf13": ppr,
                    "mf14": s + n_s + ds + n_ds + n_dtm,
                    "mf15": s + n_s + ds + n_ds,
                    "mf16": s + n_s + ds + n_ds + pr,
                    "mf17": s + n_s + ds + n_ds + npr,
                    "mf18": s + n_s + ds + n_ds + ppr
                    }

    return feature_dict


########################################################################################################################
def type_feature_dict(target_level, branch):
    '''
    Prepare type feature dict
    _, L_sort_lst, _ = create_total_L(target_level, branch)
    d = ["d_" + i for i in L_sort_lst]
    t = ["t_" + i for i in L_sort_lst]
    feature_dict = {"f1": ['s_0'] + d,
                    "f2": ['s_0'] + d + t,
                    "f3": t
                    }
    '''
    _, L_sort_lst, _ = create_total_L(target_level, branch)
    d = ["d_" + i for i in L_sort_lst]
    t = ["t_" + i for i in L_sort_lst]
    feature_dict = {"f1": ['s_0'] + d,
                    "f2": ['s_0'] + d + t,
                    "f3": t
                    }

    return feature_dict


########################################################################################################################
def print_info_dict(dict):
    print("info_dict:")
    for y in dict:
        print("\t", y, ':', dict[y])

    return


########################################################################################################################
def remove_extraBranch_and_add_bushCol(df, df_dis, tree_node_dict, child_col, parent_col, target_branch=2):
    df_anc, df_path = neuron_ancestors_and_path(df, child_col, parent_col)
    # Detect branch
    a0 = df_dis["ancestor"].tolist()
    a0 = list_freq(a0)
    a0 = [k for k, v in a0.items() if float(v) > target_branch]
    # Remove branch
    if not a0:
        pass
    else:
        a0 = sorted(a0, reverse=True)
        for p0 in a0:
            child_lst = df_dis.loc[df_dis["ancestor"]==p0, "descendant"].tolist()
            df_temp = pd.DataFrame({child_col: child_lst})
            df_temp['len'] = 0
            for c0 in child_lst:
                d0 = df_path.loc[(df_path['descendant'].isin(tree_node_dict["leaf"])) & (df_path['path'].isin([c0]))]
                d0 = pd.merge(d0, df_dis[['descendant', 'len_des_soma']], how='left', on='descendant')
                mxlen = d0['len_des_soma'].max()
                df_temp['len'] = np.where(df_temp[child_col]==c0, mxlen, df_temp['len'])

            df_temp = df_temp.sort_values('len', ascending=False)
            point_lst = df_temp[child_col].tolist()
            point_lst = point_lst[2:]
            for point in point_lst:
                path_points = df_anc.loc[df_anc['ancestor']==point, 'descendant'].tolist()
                path_points += df.loc[df['branch']==point, child_col].tolist()
                df = df.loc[~df[child_col].isin(path_points)]
                df_dis = df_dis.loc[~df_dis['descendant'].isin(path_points)]

        df = df.reset_index(drop=True)
        df_dis = df_dis.reset_index(drop=True)

    # Add bush (self + descents) col to df
    tn_lst = dict_merge_value(tree_node_dict)
    df_path = df_path.append({'path': tree_node_dict['root'][0], 'descendant': tree_node_dict['root'][0]}, ignore_index=True).sort_values(by=['descendant'])
    df_path = df_path.loc[(df_path['descendant'].isin(tn_lst)) & (df_path['path'].isin(tn_lst))]
    _df = df_path.groupby('path')['descendant'].apply(list).reset_index()
    _df = _df.rename(columns={'path': child_col, 'descendant': 'bush'})
    df = pd.merge(df, _df, how='left', on=child_col)


    return df, df_dis


########################################################################################################################
def remove_extra_branch(df, df_dis, tree_node_dict, child_col, parent_col, target_branch=2):
    if target_branch < 2:
        sys.exit("\n 'target_branch' must >= 2! Check remove_extra_branch().")

    df_anc, df_path = neuron_ancestors_and_path(df, child_col, parent_col)

    # Detect branch
    a0 = df_dis["ancestor"].tolist()
    a0 = list_freq(a0)
    a0 = [k for k, v in a0.items() if float(v) > target_branch]

    # Remove branch
    if not a0:
        pass
    else:
        a0 = sorted(a0, reverse=True)
        for p0 in a0:
            # Find len
            child_lst = df_dis.loc[df_dis["ancestor"]==p0, "descendant"].tolist()
            df_temp = df.loc[df[child_col].isin(child_lst), [child_col, "dp_level"]]
            df_temp['len'] = 0
            for c0 in child_lst:
                d0 = df_path.loc[(df_path['descendant'].isin(tree_node_dict["leaf"])) & (df_path['path'].isin([c0]))]
                d0 = pd.merge(d0, df_dis[['descendant', 'len_des_soma']], how='left', on='descendant')
                mxlen = d0['len_des_soma'].max()
                df_temp['len'] = np.where(df_temp[child_col]==c0, mxlen, df_temp['len'])
            df_temp = df_temp.sort_values('len', ascending=False)

            # Remove branch
            remove_lst = df_temp[child_col].tolist()
            remove_lst = remove_lst[2:]
            for point in remove_lst:
                path_points = df_anc.loc[df_anc['ancestor']==point, 'descendant'].tolist() + [point]
                df = df.loc[~df[child_col].isin(path_points)]
                df_dis = df_dis.loc[~df_dis['descendant'].isin(path_points)]

            # Update NC
            df["NC"] = np.where(df[child_col] == p0, 2, df["NC"])

            # Update dp_level
            dp_lst = df_temp["dp_level"].tolist()
            max_dp = max(dp_lst[:2])
            df["dp_level"] = np.where(df[child_col] == p0, max_dp, df["dp_level"])

        df = df.reset_index(drop=True)
        df_dis = df_dis.reset_index(drop=True)


    return df, df_dis


########################################################################################################################
def add_bushCol(df, tree_node_dict, child_col, parent_col, descendant_only_forkLeaf=True):
    # Add bush (self + descendants) col to df
    df_anc, df_path = neuron_ancestors_and_path(df, child_col, parent_col)

    # Add root to the df_path
    df_path = df_path.append({'path': tree_node_dict['root'][0], 'descendant': tree_node_dict['root'][0]}, ignore_index=True).sort_values(by=['descendant'])

    if descendant_only_forkLeaf:
        tn_lst = dict_merge_value(tree_node_dict)
        df_path = df_path.loc[df_path['descendant'].isin(tn_lst)]
        # df_path = df_path.loc[(df_path['descendant'].isin(tn_lst)) & (df_path['path'].isin(tn_lst))]

    _df = df_path.groupby('path')['descendant'].apply(list).reset_index()
    _df = _df.rename(columns={'path': child_col, 'descendant': 'bush'})
    df = pd.merge(df, _df, how='left', on=child_col)


    return df


########################################################################################################################
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

########################################################################################################################
def pred_mixPoint_from_prob(df, child_col="ID", label_col="label", prob_col="prob"):
    _df = df.sort_values([prob_col, child_col], ascending=[False, False]).reset_index(drop=True) # child_col ascending=F: choose larger ID as mix point
    _test = _df.loc[_df[label_col] == 1, child_col].values
    pred_lst = np.array(_df[child_col])[:len(_test)]

    df['inference'] = np.where(df[child_col].isin(pred_lst), 1, df['inference'])

    return df, pred_lst

########################################################################################################################
def turn_confusingMatrix_to_DF(cm, post_relabel=True):
    if post_relabel:
        lst = ['axon', 'dend']
    else:
        lst = ['axon', 'other']
    df = pd.DataFrame(cm, index=lst, columns=lst)
    return df

########################################################################################################################
def feature_combinations(lst):
    '''
    numbers = ["a","b","c", "d"]
    for item in it.combinations(numbers, 3):
        print(sorted(item))
    '''

    dict = {}
    for r in range(len(lst)):
        r += 1
        for idx, item in enumerate(it.combinations(lst, r)):
            k = "P"+str(r)+"-"+str(idx)
            dict[k] = sorted(item)
    return dict

########################################################################################################################
def pre_relabel(df):
    """
        Define new label to increase the quantity of data

        :param df:
            type: DataFrame
            only permit df of one neuron list (level tree)

        :return:
            type: DataFrame
            the input data with a new column: type_pre
    """

    def index_search(lst, value):
        if value == -1:
            return -1
        index = -1

        for i in range(len(lst)):
            if lst[i][0] == value:
                index = i
                break

        return index

    # load information
    df_p = df.loc[:, ['ID', 'PARENT_ID', 'T', 'NC']]
    df_lst = df_p.values.tolist()

    # modify the original type
    for i in range(len(df_lst)):
        if df_lst[i][2] == 0:
            continue
        elif df_lst[i][2] == 1:
            continue
        elif df_lst[i][2] == 2 or df_lst[i][2] // 10 == 2 or df_lst[i][2] == -2:
            df_lst[i][2] = 2
            continue
        elif df_lst[i][2] == 3 or df_lst[i][2] // 10 == 3 or df_lst[i][2] == -3:
            df_lst[i][2] = 3
            continue
        elif df_lst[i][2] == 4 or df_lst[i][2] // 10 == 4 or df_lst[i][2] == -4:
            df_lst[i][2] = 4
            continue
        else:
            print("label problem")
            print(df_lst[i][2])

    # pre-relabel
    buffer_c = []   # terminal node list index
    buffer_p = []   # parent nodes list index
    res = [0 for i in range(len(df_lst))]
    # 1. find terminal nodes, i.e. NC=0
    for i in range(len(df_lst)):
        if df_lst[i][3] == 0:
            res[i] = df_lst[i][2]   # record its type
            buffer_c.append(i)      # record its idx
    # 2.
    while len(buffer_c) != 0:
        buffer_3 = []
        # find the parent node for each terminal
        for i in buffer_c:
            buffer_p.append(index_search(df_lst, df_lst[i][1]))
        buffer = set(buffer_p)
        #
        for i in buffer:
            # if the parent node has only 1 child
            if buffer_p.count(i) == df_lst[i][3]:
                # csu todo
                location = [j for j, v in enumerate(buffer_p) if v == i]
                buffer_3 = buffer_3 + location
                buffer_c.append(i)

                buffer_2 = []
                for j in range(len(location)):
                    buffer_2.append(res[buffer_c[location[j]]])

                unlabel_count = buffer_2.count(0)
                axon_count = buffer_2.count(2)
                dendrite_count = buffer_2.count(3)
                mix_count = buffer_2.count(4)
                if unlabel_count >= 1:
                    if axon_count >= 1 and dendrite_count == 0:
                        result = 2
                    elif axon_count == 0 and dendrite_count >= 1:
                        result = 3
                    else:
                        result = 0
                elif mix_count >= 2:
                    result = 4
                elif mix_count == 1:
                    if axon_count >= 1 and dendrite_count >= 1:
                        result = 4
                    elif axon_count >= 1 and dendrite_count == 0:
                        result = 2
                    elif axon_count == 0 and dendrite_count >= 1:
                        result = 3
                    else:
                        # ktc todo what is this? (run Data_Cleaner)
                        # --> The input data should be level tree rather than original tree
                        print("problem : pre-relabeling")
                        print("problem node: %d, types: %s" % (df_lst[i][0], str(buffer_2)))
                else:
                    if axon_count == 0:
                        result = 3
                    elif dendrite_count == 0:
                        result = 2
                    else:
                        result = 4

                res[i] = result
        buffer_3.sort()
        buffer_3.reverse()
        for j in buffer_3:
            buffer_c.pop(j)
        if 0 in buffer_c:
            location = [j for j, v in enumerate(buffer_c) if v == 0]
            location.reverse()
            for j in location:
                buffer_c.pop(j)
        buffer_p = []

    # adding pre-label column
    df['type_pre'] = res

    return df


########################################################################################################################
def shape_features(df, reduced_df):
    """
        Give features of shape

        :param df:
            type: DataFrame
            only permit df of one neuron list (level tree)
        :param reduced_df:
            type: DataFrame
            only permit df of one neuron list (reduced tree)
            direct use the format of pickle

        :return:
            type: DataFrame
            the input data with new features: ratio_ortho, curvature
    """

    def index_search(lst, value):
        if value == -1:
            return -1
        index = -1

        for i in range(len(lst)):
            if lst[i][0] == value:
                index = i
                break

        return index

    def children_search(lst, value):
        index = []
        for i in range(len(lst)):
            if lst[i][1] == value:
                index.append(i)
        return index

    # load information
    reduced_list = reduced_df.tree_node_dict["fork"] + reduced_df.tree_node_dict["leaf"]
    df_p = df.loc[:, ['ID', 'PARENT_ID', 'type_pre', 'NC', 'x', 'y', 'z', 'len', 'dp_level']]
    df_lst = df_p.values.tolist()
    for i in range(len(df_lst)):
        df_lst[i][0] = int(df_lst[i][0])
    df_lst.sort(key=(lambda x: (x[0])))

    # csu todo please check the result of the following filter
    # filter of reduced tree
    if df_lst[0][0] not in reduced_list:
        reduced_list.append(df_lst[0][0])

    temp = []
    for i in range(len(reduced_list)):
        index = index_search(df_lst, reduced_list[i])
        if df_lst[index][3] != 0:
            temp.append(reduced_list[i])
        index_c = children_search(df_lst, reduced_list[i])
        for j in index_c:
            if df_lst[j][3] != 0:
                temp.append(df_lst[j][0])
    reduced_list = copy.copy(temp)
    reduced_list.sort()
    del temp

    # features
    res = [[0 for i in range(len(df_lst))] for j in range(12)]
    for i in range(len(reduced_list)):
        index = int(index_search(df_lst, reduced_list[i]))
        buffer = []
        buffer.append(index)
        index = [index]
        change = True
        while change:
            change = False
            c_index = []
            for j in index:
                temp = children_search(df_lst, df_lst[j][0])
                c_index = c_index + temp
            index = copy.copy(c_index)
            if len(c_index) != 0:
                buffer = buffer + c_index
                change = True
        if len(buffer) == 1:
            continue

        # baseline: target node
        center_x = df_lst[buffer[0]][4]
        center_y = df_lst[buffer[0]][5]
        center_z = df_lst[buffer[0]][6]

        # create the moment of inertia
        I_xx = 0
        I_xy = 0
        I_xz = 0
        I_yy = 0
        I_yz = 0
        I_zz = 0

        for j in buffer:
            if j == buffer[0]:
                continue
            I_xx += (df_lst[j][5] - center_y) ** 2 + (df_lst[j][6] - center_z) ** 2
            I_xy -= (df_lst[j][4] - center_x) * (df_lst[j][5] - center_y)
            I_xz -= (df_lst[j][4] - center_x) * (df_lst[j][6] - center_z)
            I_yy += (df_lst[j][4] - center_x) ** 2 + (df_lst[j][6] - center_z) ** 2
            I_yz -= (df_lst[j][5] - center_y) * (df_lst[j][6] - center_z)
            I_zz += (df_lst[j][4] - center_x) ** 2 + (df_lst[j][5] - center_y) ** 2
        I_value, I_vector = np.linalg.eig(np.array([[I_xx, I_xy, I_xz],
                                                    [I_xy, I_yy, I_yz],
                                                    [I_xz, I_yz, I_zz]]))
        index = I_value.argmin()
        p_max = 0
        v_max = 0
        for j in buffer:
            if j == buffer[0]:
                continue
            v = np.array([df_lst[j][4] - center_x, df_lst[j][5] - center_y, df_lst[j][6] - center_z])
            l = np.dot(v, v)
            v_p = abs(np.dot(v, I_vector[:][index]))
            if v_p > p_max:
                p_max = v_p
            l = l - v_p**2
            l = math.sqrt(l)
            if l > v_max:
                v_max = l
        for j in buffer:
            #res[j][0] = p_max
            #res[j][1] = v_max
            res[0][j] = v_max / (p_max + v_max)

        p_max = [0, 0, 0]
        p_min = [0, 0, 0]
        for j in buffer:
            if j == buffer[0]:
                continue
            v = np.array([df_lst[j][4] - center_x, df_lst[j][5] - center_y, df_lst[j][6] - center_z])
            for k in range(I_vector.shape[1]):
                v_p = np.dot(v, I_vector[:][k])
                if v_p > p_max[k]:
                    p_max[k] = v_p
                if v_p < p_min[k]:
                    p_min[k] = v_p
            for k in range(len(p_max)):
                p_max[k] = p_max[k] - p_min[k]
            p_max.sort()
        for j in buffer:
            #res[j][0] = p_max
            #res[j][1] = v_max
            res[1][j] = p_max[-1] / (p_max[0] + p_max[-1])

        # baseline: center of mass
        center_x = 0
        center_y = 0
        center_z = 0
        for j in buffer:
            center_x += df_lst[j][4]/len(buffer)
            center_y += df_lst[j][5]/len(buffer)
            center_z += df_lst[j][6]/len(buffer)

        # create the moment of inertia
        I_xx = 0
        I_xy = 0
        I_xz = 0
        I_yy = 0
        I_yz = 0
        I_zz = 0

        for j in buffer:
            if j == buffer[0]:
                continue
            I_xx += (df_lst[j][5] - center_y) ** 2 + (df_lst[j][6] - center_z) ** 2
            I_xy -= (df_lst[j][4] - center_x) * (df_lst[j][5] - center_y)
            I_xz -= (df_lst[j][4] - center_x) * (df_lst[j][6] - center_z)
            I_yy += (df_lst[j][4] - center_x) ** 2 + (df_lst[j][6] - center_z) ** 2
            I_yz -= (df_lst[j][5] - center_y) * (df_lst[j][6] - center_z)
            I_zz += (df_lst[j][4] - center_x) ** 2 + (df_lst[j][5] - center_y) ** 2
        I_value, I_vector = np.linalg.eig(np.array([[I_xx, I_xy, I_xz],
                                                    [I_xy, I_yy, I_yz],
                                                    [I_xz, I_yz, I_zz]]))

        characteristic_length = []
        for j in range(len(I_value)):
            characteristic_length.append(math.sqrt(I_value[j] / len(buffer)))
        characteristic_length.sort()
        temp = characteristic_length[0]/characteristic_length[-1]
        for j in range(len(buffer)):
            res[11][j] = temp

        index = I_value.argmin()
        p_max = 0
        v_max = 0
        for j in buffer:
            if j == buffer[0]:
                continue
            v = np.array([df_lst[j][4] - center_x, df_lst[j][5] - center_y, df_lst[j][6] - center_z])
            l = np.dot(v, v)
            v_p = np.dot(v, I_vector[index])
            if v_p > p_max:
                p_max = v_p
            l = l - v_p**2
            l = math.sqrt(l)
            if l > v_max:
                v_max = l
        for j in buffer:
            # res[j][0] = p_max
            # res[j][1] = v_max
            res[2][j] = v_max / (p_max + v_max)

        p_max = [0, 0, 0]
        p_min = [0, 0, 0]
        for j in buffer:
            if j == buffer[0]:
                continue
            v = np.array([df_lst[j][4] - center_x, df_lst[j][5] - center_y, df_lst[j][6] - center_z])
            for k in range(I_vector.shape[1]):
                v_p = np.dot(v, I_vector[:][k])
                if v_p > p_max[k]:
                    p_max[k] = v_p
                if v_p < p_min[k]:
                    p_min[k] = v_p
            for k in range(len(p_max)):
                p_max[k] = p_max[k] - p_min[k]
            p_max.sort()
        for j in buffer:
            # res[j][0] = p_max
            # res[j][1] = v_max
            res[3][j] = p_max[-1] / (p_max[0] + p_max[-1])


        # Number in the range (under)
        temp_num = 0
        temp_curve = 0

        for j in buffer:
            if j == buffer[0]:
                continue
            length = (df_lst[buffer[0]][4]-df_lst[j][4])**2+(df_lst[buffer[0]][5]-df_lst[j][5])**2+(df_lst[buffer[0]][6]-df_lst[j][6])**2
            length = math.sqrt(length)

            temp_num += 1
            length_2 = df_lst[j][7]
            index_p = index_search(df_lst, df_lst[j][1])
            while index_p != buffer[0]:
                length_2 += df_lst[index_p][7]
                index_p = index_search(df_lst, df_lst[index_p][1])
            try:
                temp_curve += length_2/length
            except:
                pass
        '''
        for j in buffer:
            res[j][3] = temp_num
            res[j][4] = temp_curve
        '''
        for j in buffer:
            res[4][j] = temp_curve
        if temp_num != 0:
            temp_curve = temp_curve / temp_num
        else:
            temp_curve = 0
        for j in buffer:
            res[5][j] = temp_curve

        # Number in the range (under)
        dis_max = 0

        #for j in buffer:
        #    if j == buffer[0]:
        #        continue
        #    dis = (df_lst[buffer[0]][4] - df_lst[j][4]) ** 2 + (df_lst[buffer[0]][5] - df_lst[j][5]) ** 2 + (
        #                df_lst[buffer[0]][6] - df_lst[j][6]) ** 2
        #    dis = math.sqrt(dis)
        #    if dis > dis_max:
        #        dis_max = dis
        dis_max = math.pow(characteristic_length[0]*characteristic_length[1]*characteristic_length[2], 1.0/3)

        length = 0
        for j in buffer:
            if j == buffer[0]:
                continue
            length += df_lst[j][7]

        length2 = length + df_lst[buffer[0]][7]

        temp_curve = 0
        try:
            temp_curve += length / dis_max
        except:
            temp_curve = 0
        for j in buffer:
            res[6][j] = temp_curve
        del temp_curve

        temp_curve = 0
        try:
            temp_curve += length2 / dis_max
        except:
            temp_curve = 0
        for j in buffer:
            res[7][j] = temp_curve

        terminal_points = 0
        index = buffer[0]
        c_index = children_search(df_lst, df_lst[index][0])
        while len(c_index) != 0:
            index = []
            for i in c_index:
                temp = children_search(df_lst, df_lst[i][0])
                if len(temp) == 0:
                    terminal_points += 1
                else:
                    index = index + temp
            c_index = index

        # convex hull
        temp_lst = []
        if len(buffer) > 3:
            for j in buffer:
                temp_lst.append(df_lst[j][4:7])
            try:
                cv = ConvexHull(temp_lst)
                vol = cv.volume
                ave_vol_length = vol / length
                ave_vol_terminals = vol / terminal_points

                for j in buffer:
                    res[8][j] = vol
                    res[9][j] = ave_vol_length
                    res[10][j] = ave_vol_terminals
            except:
                print("Hull Problem")
                print(buffer)
                pass

    df['ratio_ortho'] = res[0]
    df['ratio2_ortho'] = res[1]
    df['ratio_com'] = res[2]
    df['ratio2_com'] = res[3]
    df['curvature_superposition'] = res[4]
    df['curvature_ave'] = res[5]
    df['curvature_r'] = res[6]
    df['curvature'] = res[7]
    df['volume'] = res[8]
    df['ave_volume_length'] = res[9]
    df['ave_volume_terminals'] = res[10]
    df['aspect_ratio'] = res[11]

    # ratio children
    res = [0 for i in range(len(df_lst))]
    for i in range(len(df_lst)):
        index_c = children_search(df_lst, df_lst[i][0])
        if len(index_c) == 0:
            res[i] = 0
            continue
        temp = 0
        temp_s = 0
        for j in index_c:
            temp_s += df_lst[j][7]
            if df_lst[j][7] > temp:
                temp = df_lst[j][7]
        if temp_s != 0:
            res[i] = temp/temp_s
        else:
            res[i] = 0
    df['ratio_children'] = res

    return df


########################################################################################################################
def non_relabel(df):
    def index_search(lst, value):
        if value == -1:
            return -1
        index = -1

        for i in range(len(lst)):
            if lst[i][1] == value:
                index = i
                break

        return index

    def children_search(lst, value):
        index = []
        for i in range(len(lst)):
            if lst[i][2] == value:
                index.append(i)
        return index

    def relabel_process(lst):
        # Step1. fill terminals
        for j in range(len(lst)):
            if lst[j][3] == 0 and lst[j][4] == 0:
                index = index_search(lst, lst[j][2])
                while True:
                    if lst[index][4] != 0:
                        lst[j][4] = lst[index][4]
                        break
                    index = index_search(lst, lst[index][2])
                    if index == -1:
                        break
        return lst

    def threshold_process(lst):
        th = 0.5
        for j in range(len(lst)):
            if math.isnan(lst[j][4]):
                lst[j].pop(-1)
                lst[j].append(0)
                continue

            if lst[j][4] >= th:
                lst[j][4] = 2
            elif lst[j][4] < 1 - th:
                lst[j][4] = 3
            else:
                lst[j][4] = 0
        return lst

    # load information
    df.sort_values(by=['nrn', 'ID'])
    df_p = df.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'prob']]
    df_lst = df_p.values.tolist()

    # neuron name
    file_name = []
    point_s = []
    for i in range(len(df_lst)):
        if df_lst[i][0] not in file_name:
            point_s.append(i)
            file_name.append(df_lst[i][0])

    for i in range(len(file_name)):
        # replacing by the method of terminals
        lst1 = []
        for j in range(point_s[i], len(df_lst)):
            if df_lst[j][0] != file_name[i]:
                break
            lst1.append(df_lst[j])

        # Calculate
        lst2 = threshold_process(lst1)
        lst2 = relabel_process(lst2)

        for j in range(len(lst2)):
            df_lst[point_s[i] + j][4] = lst2[j][4]

    res = []
    for i in range(len(df_lst)):
        res.append(df_lst[i][4])
    df["type_non_relabel"] = res

    return df


########################################################################################################################
def post_relabel(df, threshold=0.8, reduce_value=0.05):
    """
        Give the prediction after relabeling

        :param df:
            type: DataFrame
            df of all neurons (level tree)
            need to be sorted by nrn ID
            what you should add:
                nrn (type:str): each node should have a corresponding neuron
                PROBABILTY (type:list): probabilties (to be axon) predicted by machines
        :param threshold:
            type: float
            the threshold of classifying neuronal polarity

        :return:
            type: DataFrame
            the input data with a new column: type_post
    """

    def index_search(lst, value):
        if value == -1:
            return -1
        index = -1

        for i in range(len(lst)):
            if lst[i][1] == value:
                index = i
                break

        return index

    def children_search(lst, value):
        index = []
        for i in range(len(lst)):
            if lst[i][2] == value:
                index.append(i)
        return index

    def relabel_process(lst):
        for j in range(len(lst)):
            if lst[j][3] == 0 and lst[j][4] == 0:
                index = index_search(lst, lst[j][2])
                while True:
                    if lst[index][4] != 0:
                        lst[j][4] = lst[index][4]
                        break
                    index = index_search(lst, lst[index][2])
                    if index == -1:
                        break

        buffer_c = []
        buffer_p = []

        for j in range(len(lst)):
            if lst[j][3] == 0:
                buffer_c.append(j)

        while len(buffer_c) != 0:
            buffer_3 = []
            for j in buffer_c:
                buffer_p.append(index_search(lst, lst[j][2]))
            buffer = set(buffer_p)
            for j in buffer:
                if buffer_p.count(j) == lst[j][3]:

                    location = [k for k, v in enumerate(buffer_p) if v == j]
                    buffer_3 = buffer_3 + location
                    buffer_c.append(j)

                    buffer_2 = []
                    for k in range(len(location)):
                        buffer_2.append(lst[buffer_c[location[k]]][4])

                    dendrite_count = buffer_2.count(3)
                    axon_count = buffer_2.count(2)
                    mix_count = buffer_2.count(4)

                    if mix_count > 1:
                        result = 4
                    else:
                        if dendrite_count != 0 and axon_count == 0:
                            result = 3
                        elif dendrite_count == 0 and axon_count != 0:
                            result = 2
                        elif dendrite_count != 0 and axon_count != 0:
                            result = 4
                        else:
                            result = 0

                    lst[j][4] = result

            buffer_3.sort()
            buffer_3.reverse()
            for k in buffer_3:
                buffer_c.pop(k)
            if 0 in buffer_c:
                location = [k for k, v in enumerate(buffer_c) if v == 0]
                location.reverse()
                for k in location:
                    buffer_c.pop(k)
            buffer_p = []

        for j in range(len(lst)):
            c_index = children_search(lst, lst[j][1])
            for k in c_index:
                if lst[k][-1] == 0:
                    lst[k][-1] = lst[j][-1]
        return lst

    def threshold_process(lst, th):
        for j in range(len(lst)):
            if math.isnan(lst[j][4]):
                lst[j].pop(-1)
                lst[j].append(0)
                continue

            if lst[j][4] >= th:
                lst[j][4] = 2
            elif lst[j][4] < 1 - th:
                lst[j][4] = 3
            else:
                lst[j][4] = 0
        return lst

    # load information
    df.sort_values(by=['nrn', 'ID'])
    df_p = df.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'prob']]
    df_lst = df_p.values.tolist()

    # neuron name
    file_name = []
    point_s = []
    for i in range(len(df_lst)):
        if df_lst[i][0] not in file_name:
            point_s.append(i)
            file_name.append(df_lst[i][0])

    for i in range(len(file_name)):

        # filter 1: before relabel
        threshold_i = threshold
        while threshold_i != 0.5:
            dendrite_like = False
            axon_like = False
            for j in range(point_s[i], len(df_lst)):
                if df_lst[j][0] != file_name[i]:
                    break
                if df_lst[j][4] >= threshold_i:
                    axon_like = True
                elif df_lst[j][4] < 1 - threshold_i:
                    dendrite_like = True
                if axon_like and dendrite_like:
                    break
            if axon_like and dendrite_like:
                break
            else:
                threshold_i = threshold_i - reduce_value
                if threshold_i > 0.5:
                    pass
                else:
                    threshold_i = 0.5

        # replacing by the method of terminals
        lst1 = []
        for j in range(point_s[i], len(df_lst)):
            if df_lst[j][0] != file_name[i]:
                break
            lst1.append(df_lst[j])

        # Calculate
        lst2 = threshold_process(lst1, threshold_i)
        lst2 = relabel_process(lst2)

    #filter 2: after relabel
        while threshold_i != 0.5:
            dendrite_like = False
            axon_like = False
            for j in range(len(lst2)):
                if lst2[j][4] == 2:
                    axon_like = True
                elif lst2[j][4] == 3:
                    dendrite_like = True
            if axon_like and dendrite_like:
                break
            else:
                threshold_i -= reduce_value
                if threshold_i > 0.5:
                    lst2 = threshold_process(lst1, threshold_i)
                    lst2 = relabel_process(lst2)
                else:
                    threshold_i = 0.5
                    lst2 = threshold_process(lst1, threshold_i)
                    lst2 = relabel_process(lst2)

        for j in range(len(lst2)):
            df_lst[point_s[i] + j][4] = lst2[j][4]


    # check if both axon and dendrite exist
    bad_case = []
    for i in range(len(file_name)):
        lst = []
        for j in range(point_s[i], len(df_lst)):
            if df_lst[j][0] != file_name[i]:
                break
            lst.append(df_lst[j])

        dendrite_like = False
        axon_like = False
        for j in range(len(lst)):
            if lst[j][4] == 2:
                axon_like = True
            elif lst[j][4] == 3:
                dendrite_like = True
        if axon_like and dendrite_like:
            pass
        else:
            bad_case.append(i)

    lst = ""
    if len(bad_case) == 0:
        lst = "None"
    else:
        lst = lst + file_name[bad_case[0]]
        for i in range(1, len(bad_case)):
            lst += " " + file_name[bad_case[i]]
    print("still bad : ", lst)

    res = []
    for i in range(len(df_lst)):
        res.append(df_lst[i][4])
    df["type_post"] = res

    return df


########################################################################################################################
def hyper_relabel(lst_of_df, threshold=0.8, reduce_th=0.05):
    """
        Give the better prediction of various models

        :param lst_of_df:
            type: list
            [ [name_of_model1, model1], [name_of_model1, model1], ... ]
            name_of_model: str
                --> just type the name you want
            model: DataFrame
                --> which should contain the following columns: 'nrn', 'ID', 'PARENT_ID', 'NC', 'prob'
        :return:

    """

    def index_search(lst, value):
        if value == -1:
            return -1
        index = -1

        for i in range(len(lst)):
            if lst[i][1] == value:
                index = i
                break

        return index

    def children_search(lst, value):
        index = []
        for i in range(len(lst)):
            if lst[i][2] == value:
                index.append(i)
        return index

    def relabel_process(nrn_lst, point_start):
        # Step1. fill terminals
        for i in range(len(point_start)):

            if i != len(point_start) - 1:
                start = point_start[i]
                end = point_start[i+1]
            else:
                start = point_start[i]
                end = len(nrn_lst)
            lst = []
            for j in range(start, end):
                lst.append(copy.copy(nrn_lst[j]))
            for j in range(len(lst)):
                if lst[j][3] == 0 and lst[j][4] == 0:
                    index = index_search(lst, lst[j][2])
                    while True:
                        if lst[index][4] != 0:
                            lst[j][4] = lst[index][4]
                            break
                        index = index_search(lst, lst[index][2])
                        if index == -1:
                            break

            buffer_c = []
            buffer_p = []

            for j in range(len(lst)):
                if lst[j][3] == 0:
                    buffer_c.append(j)

            while len(buffer_c) != 0:
                buffer_3 = []
                for j in buffer_c:
                    buffer_p.append(index_search(lst, lst[j][2]))
                buffer = set(buffer_p)
                for j in buffer:
                    if buffer_p.count(j) == lst[j][3]:

                        location = [k for k, v in enumerate(buffer_p) if v == j]
                        buffer_3 = buffer_3 + location
                        buffer_c.append(j)

                        buffer_2 = []
                        for k in range(len(location)):
                            buffer_2.append(lst[buffer_c[location[k]]][4])

                        dendrite_count = buffer_2.count(3)
                        axon_count = buffer_2.count(2)
                        mix_count = buffer_2.count(4)

                        if mix_count > 1:
                            result = 4
                        else:
                            if dendrite_count != 0 and axon_count == 0:
                                result = 3
                            elif dendrite_count == 0 and axon_count != 0:
                                result = 2
                            elif dendrite_count != 0 and axon_count != 0:
                                result = 4
                            else:
                                result = 0

                        lst[j][4] = result

                buffer_3.sort()
                buffer_3.reverse()
                for k in buffer_3:
                    buffer_c.pop(k)
                if 0 in buffer_c:
                    location = [k for k, v in enumerate(buffer_c) if v == 0]
                    location.reverse()
                    for k in location:
                        buffer_c.pop(k)
                buffer_p = []

            for j in range(len(lst)):
                c_index = children_search(lst, lst[j][1])
                for k in c_index:
                    if lst[k][-1] == 0:
                        lst[k][-1] = lst[j][-1]
            for j in range(len(lst)):
                nrn_lst[start + j][4] = lst[j][4]
        return nrn_lst

    def relabel_process2(lst):
        # Step1. fill unlabel
        for j in range(len(lst)):
            if lst[j][4] != 0:
                index = index_search(lst, lst[j][2])
                while lst[index][4] == 0:
                    lst[index][4] = lst[j][4]
                    index = index_search(lst, lst[index][2])
                    if index == -1:
                        break

        # Step2. fill terminals
        for j in range(len(lst)):
            if lst[j][3] == 0 and lst[j][4] == 0:
                index = index_search(lst, lst[j][2])
                while True:
                    if lst[index][4] != 0:
                        lst[j][4] = lst[index][4]
                        break
                    index = index_search(lst, lst[index][2])
                    if index == -1:
                        break

        buffer_c = []
        buffer_p = []

        for j in range(len(lst)):
            if lst[j][3] == 0:
                buffer_c.append(j)

        while len(buffer_c) != 0:
            buffer_3 = []
            for j in buffer_c:
                buffer_p.append(index_search(lst, lst[j][2]))
            buffer = set(buffer_p)
            for j in buffer:
                if buffer_p.count(j) == lst[j][3]:

                    location = [k for k, v in enumerate(buffer_p) if v == j]
                    buffer_3 = buffer_3 + location
                    buffer_c.append(j)

                    buffer_2 = []
                    for k in range(len(location)):
                        buffer_2.append(lst[buffer_c[location[k]]][4])

                    dendrite_count = buffer_2.count(3)
                    axon_count = buffer_2.count(2)
                    mix_count = buffer_2.count(4)

                    if mix_count > 1:
                        result = 4
                    else:
                        if dendrite_count != 0 and axon_count == 0:
                            result = 3
                        elif dendrite_count == 0 and axon_count != 0:
                            result = 2
                        elif dendrite_count != 0 and axon_count != 0:
                            result = 4
                        else:
                            result = 0

                    lst[j][4] = result

            buffer_3.sort()
            buffer_3.reverse()
            for k in buffer_3:
                buffer_c.pop(k)
            if 0 in buffer_c:
                location = [k for k, v in enumerate(buffer_c) if v == 0]
                location.reverse()
                for k in location:
                    buffer_c.pop(k)
            buffer_p = []

        for j in range(len(lst)):
            c_index = children_search(lst, lst[j][1])
            for k in c_index:
                if lst[k][-1] == 0:
                    lst[k][-1] = lst[j][-1]
        return lst

    def threshold_process(lst, th):
        lst = copy.deepcopy(lst)
        for j in range(len(lst)):
            if math.isnan(lst[j][4]):
                lst[j].pop(-1)
                lst[j].append(0)
                continue

            if th <= lst[j][4] <= 1:
                lst[j][4] = 2
            elif 0 <= lst[j][4] < 1 - th:
                lst[j][4] = 3
            else:
                lst[j][4] = 0
        return lst

    def fill_terminals(lst):
        for j in range(len(lst)):
            if lst[j][3] == 0:
                if lst[j][4] == "nan" or lst[j][4] == -1:
                    index = index_search(lst, lst[j][2])
                    while True:
                        if lst[index][4] != 0:
                            lst[j][4] = lst[index][4]
                            break
                        index = index_search(lst, lst[index][2])
                        if index == -1:
                            break
        return lst

    def hyper_process(lst1, lst2, th, name1, name2, point_lst, re_th):
        #lst1 = fill_terminals(lst1)
        #lst2 = fill_terminals(lst2)
        lst3 = copy.deepcopy(lst1)

        # hyper relabel process
        hyper_cm = [[0 for i in range(3)] for j in range(3)]
        for i in range(len(point_lst)):
            th1 = th
            th2 = th
            if i == len(point_lst)-1:
                start_p = point_lst[i]
                end_p = len(lst1)
            else:
                start_p = point_lst[i]
                end_p = point_lst[i+1]

            # threshold process
            while True:
                temp_cm = [[0 for j in range(3)] for k in range(3)]

                for j in range(start_p, end_p):
                    if 1 >= lst1[j][4] > th1:
                        if 1 >= lst2[j][4] > th2:
                            temp_cm[0][2] += 1
                            lst3[j][4] = 2
                        elif th2 >= lst2[j][4] >= 1-th2:
                            temp_cm[0][1] += 1
                            lst3[j][4] = 2
                        elif 1-th2 > lst2[j][4] >= 0:
                            temp_cm[0][0] += 1
                            lst3[j][4] = 0
                        else:
                            print(lst2[j][4])
                            print("hyper_process_problem1")
                    elif th1 >= lst1[j][4] >= 1-th1:
                        if 1 >= lst2[j][4] > th2:
                            temp_cm[1][2] += 1
                            lst3[j][4] = 2
                        elif th2 >= lst2[j][4] >= 1-th2:
                            temp_cm[1][1] += 1
                            lst3[j][4] = 0
                        elif 1-th2 > lst2[j][4] >= 0:
                            temp_cm[1][0] += 1
                            lst3[j][4] = 3
                        else:
                            print(lst2[j][4])
                            print("hyper_process_problem2")
                    elif 1-th1 > lst1[j][4] >= 0:
                        if 1 >= lst2[j][4] > th2:
                            temp_cm[2][2] += 1
                            lst3[j][4] = 0
                        elif th2 >= lst2[j][4] >= 1-th2:
                            temp_cm[2][1] += 1
                            lst3[j][4] = 3
                        elif 1-th2 > lst2[j][4] >= 0:
                            temp_cm[2][0] += 1
                            lst3[j][4] = 3
                        else:
                            print(lst2[j][4])
                            print("hyper_process_problem3")
                    else:
                        print(lst1[j][4])
                        print("hyper_process_problem4")

                # check thresholds of models
                threshold_adjust = False
                if (temp_cm[0][0] + temp_cm[0][1] + temp_cm[0][2] == 0) or (temp_cm[2][0] + temp_cm[2][1] + temp_cm[2][2] == 0):
                    if th1 > 0.5:
                        th1 -= re_th
                        if th1 < 0.5:
                            th1 = 0.5
                        threshold_adjust = True
                    elif th1 == 0.5:
                        pass
                    else:
                        print("hyper_process_threshold_problem")

                if (temp_cm[0][0] + temp_cm[1][0] + temp_cm[2][0] == 0) or (temp_cm[0][2] + temp_cm[1][2] + temp_cm[2][2] == 0):
                    if th2 > 0.5:
                        th2 -= re_th
                        if th2 < 0.5:
                            th2 = 0.5
                        threshold_adjust = True
                    elif th2 == 0.5:
                        pass
                    else:
                        print("hyper_process_threshold_problem")

                if not threshold_adjust:
                    break
            for j in range(3):
                for k in range(3):
                    hyper_cm[j][k] += temp_cm[j][k]

        print("==========================================================")
        print("model1: ", name1)
        print("model2: ", name2)
        print("==========================================================")
        print('| model1 \\ model2 |  Dendrite  |  Unknown   |  Axon      |')
        print('----------------------------------------------------------')
        print("|  {:<12s}   |  {:<9d} |  {:<9d} |  {:<9d} |".format("Axon", hyper_cm[0][0],
                                                                    hyper_cm[0][1], hyper_cm[0][2]))
        print("|  {:<12s}   |  {:<9d} |  {:<9d} |  {:<9d} |".format("Unknown", hyper_cm[1][0],
                                                                    hyper_cm[1][1], hyper_cm[1][2]))
        print("|  {:<12s}   |  {:<9d} |  {:<9d} |  {:<9d} |".format("Dendrite", hyper_cm[2][0],
                                                                    hyper_cm[2][1], hyper_cm[2][2]))
        print("==========================================================")
        return lst3, hyper_cm

    def hyper_process_neuron(lst1, lst2, th, name1, name2, point_lst, re_th):
        #lst1 = fill_terminals(lst1)
        #lst2 = fill_terminals(lst2)
        lst3 = copy.deepcopy(lst1)

        # hyper relabel process
        hyper_cm = [[0 for i in range(3)] for j in range(3)]
        for i in range(len(point_lst)):
            th1 = th
            if i == len(point_lst) - 1:
                start_p = point_lst[i]
                end_p = len(lst1)
            else:
                start_p = point_lst[i]
                end_p = point_lst[i + 1]

            # threshold process
            score_lst1 = 0
            score_lst2 = 0
            for j in range(start_p, end_p):
                score_lst1 += (2*abs(lst1[j][4]-0.5))**2
                score_lst2 += (2*abs(lst2[j][4]-0.5))**2
            if score_lst1 > score_lst2:
                while True:
                    Axon_like = False
                    Dendrite_like = False

                    for j in range(start_p, end_p):
                        if 1 >= lst1[j][4] > th1:
                            lst3[j][4] = 2
                            Axon_like = True
                        elif th1 >= lst1[j][4] >= 1 - th1:
                            lst3[j][4] = 0
                        elif 1 - th1 > lst1[j][4] >= 0:
                            lst3[j][4] = 3
                            Dendrite_like = True
                        else:
                            print("hyper_process_neuron_problem1")

                    if Axon_like and Dendrite_like:
                        break
                    else:
                        if th1 > 0.5:
                            th1 -= re_th
                            if th1 < 0.5:
                                th1 = 0.5
                                continue
                        elif th1 == 0.5:
                            break
                        else:
                            print("hyper_process_threshold_problem")
            else:
                while True:
                    Axon_like = False
                    Dendrite_like = False

                    for j in range(start_p, end_p):
                        if 1 >= lst2[j][4] > th1:
                            lst3[j][4] = 2
                            Axon_like = True
                        elif th1 >= lst2[j][4] >= 1 - th1:
                            lst3[j][4] = 0
                        elif 1 - th1 > lst2[j][4] >= 0:
                            lst3[j][4] = 3
                            Dendrite_like = True
                        else:
                            print("hyper_process_neuron_problem1")

                    if Axon_like and Dendrite_like:
                        break
                    else:
                        if th1 > 0.5:
                            th1 -= re_th
                            if th1 < 0.5:
                                th1 = 0.5
                                continue
                        elif th1 == 0.5:
                            break
                        else:
                            print("hyper_process_threshold_problem")

        return lst3

    with open(os.getcwd() + "/final.csv", newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        rows.pop(0)

    neuron_set = {}
    for i in range(len(rows)):
        if rows[i][1] not in neuron_set.keys():
            neuron_set[rows[i][1]] = [rows[i][0]]
            continue
        if rows[i][1] in neuron_set.keys():
            neuron_set[rows[i][1]].append(rows[i][0])

    n_set = list(neuron_set.keys())

    # load information and check the length of neuron_df
    models = locals()
    model_lst = []
    for i in range(len(lst_of_df)):
        df = lst_of_df[i][1]
        df.sort_values(by=['nrn', 'ID'])
        models["model_" + str(i+1) + "_lst"] = df.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'prob']].values.tolist()
        model_lst.append("model_" + str(i+1) + "_lst")
        if i == 0:
            length = len(models["model_" + str(i+1) + "_lst"])
        else:
            if len(models["model_" + str(i+1) + "_lst"]) != length:
                print("length problem")
                sys.exit()

    # check neuron name and the coincidence between models
    file_name = []
    point_s = []
    for i in range(len(models["model_1_lst"])):
        if models["model_1_lst"][i][0] not in file_name:
            point_s.append(i)
            file_name.append(models["model_1_lst"][i][0])
    for i in range(1, len(model_lst)):
        file_temp = []
        for j in range(len(models[model_lst[i]])):
            if models[model_lst[i]][j][0] not in file_temp:
                file_temp.append(models[model_lst[i]][j][0])
                if j != point_s[len(file_temp)-1]:
                    print("coincidence problem")
                    sys.exit()

    lst = [i for i in range(len(model_lst))]
    comparison = list(it.combinations(lst, 2))
    mix_models = {}
    output_csv = []
    # hyper-relabel structure
    # threshold combination process
    print("Level: Nodes")
    print("Hyper Relabel: Combination Process")
    for num in range(len(comparison)):
        index1 = comparison[num][0]
        index2 = comparison[num][1]
        name = "model" + "_" + str(index1) + "_" + str(index2) + "_lst"
        mix_models[name], hy_cm = hyper_process(models[model_lst[index1]], models[model_lst[index2]], threshold,
                                                lst_of_df[index1][0], lst_of_df[index2][0], point_s, reduce_th)

        # output csv
        output_csv += [[] for i in range(11)]
        output_csv[num*11] += ["Model1: ", lst_of_df[index1][0]]
        output_csv[num*11+1] += ["Model2: ", lst_of_df[index2][0]]
        output_csv[num*11+2] += ["", "", "", "", ""]
        output_csv[num*11+3] += ["Confusion", "Matrix", "", "", ""]
        output_csv[num*11+4] += ["Model1 \\ Model2", "Dendrite", "Undefinde", "Axon", ""]
        output_csv[num*11+5] += ["Axon", hy_cm[0][0], hy_cm[0][1], hy_cm[0][2], ""]
        output_csv[num*11+6] += ["Undefinde", hy_cm[1][0], hy_cm[1][1], hy_cm[1][2], ""]
        output_csv[num*11+7] += ["Dendrite", hy_cm[2][0], hy_cm[2][1], hy_cm[2][2], ""]
        output_csv[num*11+8] += ["", "", "", "", ""]
        output_csv[num*11+9] += ["", "", "", "", ""]
        output_csv[num*11+10] += [""]

    # relabel process
    print("Hyper Relabel: Post-Relabel Process")
    keys = list(mix_models.keys())
    for key in keys:
        mix_models[key] = relabel_process(mix_models[key], point_s)
        mix_models[key] = pd.DataFrame(mix_models[key], columns=['nrn', 'ID', 'PARENT_ID', 'NC', 'type_post'])
        mix_models[key] = mix_models[key].loc[:, ['nrn', 'ID', 'type_post']]

    # check result Nodes
    print("Hyper Relabel: Print Results")
    df = lst_of_df[0][1]
    df.sort_values(by=['nrn', 'ID'])
    id_check = df.loc[:, ['nrn', 'ID', 'NC', 'type_pre']]
    for num in range(len(comparison)):
        index1 = comparison[num][0]
        index2 = comparison[num][1]
        name = "model" + "_" + str(index1) + "_" + str(index2) + "_lst"
        res = pd.merge(id_check, mix_models[name], how='left', on=['nrn', 'ID'])
        res = res.values.tolist()
        cm = [[0, 0], [0, 0]]
        lost_points = 0
        for i in range(len(res)):
            if res[i][2] == 0:
                if res[i][3] == 2:
                    if res[i][4] == 2:
                        cm[0][0] += 1
                    elif res[i][4] == 3:
                        cm[0][1] += 1
                    else:
                        lost_points += 1
                elif res[i][3] == 3:
                    if res[i][4] == 2:
                        cm[1][0] += 1
                    elif res[i][4] == 3:
                        cm[1][1] += 1
                    else:
                        lost_points += 1

        if cm[0][0]+cm[1][0] != 0:
            precision_axon = 100*cm[0][0]/(cm[0][0]+cm[1][0])
            precision_axon = "{:>.2f}".format(precision_axon) + "%"
        else:
            precision_axon = "nan"
        if cm[1][1]+cm[0][1] != 0:
            precision_dendrite = 100*cm[1][1]/(cm[1][1]+cm[0][1])
            precision_dendrite = "{:>.2f}".format(precision_dendrite) + "%"
        else:
            precision_dendrite = "nan"
        if cm[0][0]+cm[0][1] != 0:
            recall_axon = 100*cm[0][0]/(cm[0][0]+cm[0][1])
            recall_axon = "{:>.2f}".format(recall_axon) + "%"
        else:
            recall_axon = "nan"
        if cm[1][1]+cm[1][0] != 0:
            recall_dendrite = 100*cm[1][1]/(cm[1][1]+cm[1][0])
            recall_dendrite = "{:>.2f}".format(recall_dendrite) + "%"
        else:
            recall_dendrite = "nan"
        accuracy = 100*(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])

        print("==========================================================")
        print("Neuron Set: total")
        print("model1: " + lst_of_df[index1][0])
        print("model2: " + lst_of_df[index2][0])
        print("lost points: ", lost_points)
        print('==========================================================')
        print('| Real \\ Pred |   Axon   | Dendrite |')
        print('----------------------------------------------------------')
        print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Axon", cm[0][0], cm[0][1]))
        print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Dendrite", cm[1][0], cm[1][1]))
        print("==========================================================")
        print('==========================================================')
        print('|             | Precision |  Recall   |')
        print('----------------------------------------------------------')
        print("|{:^13s}|{:^11s}|{:^11s}|".format("Axon", precision_axon, recall_axon))
        print("|{:^13s}|{:^11s}|{:^11s}|".format("Dendrite", precision_dendrite, recall_dendrite))
        print("Accuracy: {:<.2f}%".format(accuracy))
        print("==========================================================\n")

        for i in n_set:
            cm = [[0, 0], [0, 0]]
            lost_points = 0
            for j in range(len(res)):
                if res[j][0] in neuron_set[i]:
                    if res[j][2] == 0:
                        if res[j][3] == 2:
                            if res[j][4] == 2:
                                cm[0][0] += 1
                            elif res[j][4] == 3:
                                cm[0][1] += 1
                            else:
                                lost_points += 1
                        elif res[j][3] == 3:
                            if res[j][4] == 2:
                                cm[1][0] += 1
                            elif res[j][4] == 3:
                                cm[1][1] += 1
                            else:
                                lost_points += 1
            if cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == 0:
                continue

            if cm[0][0] + cm[1][0] != 0:
                precision_axon = 100 * cm[0][0] / (cm[0][0] + cm[1][0])
                precision_axon = "{:>.2f}".format(precision_axon) + "%"
            else:
                precision_axon = "nan"
            if cm[1][1] + cm[0][1] != 0:
                precision_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[0][1])
                precision_dendrite = "{:>.2f}".format(precision_dendrite) + "%"
            else:
                precision_dendrite = "nan"
            if cm[0][0] + cm[0][1] != 0:
                recall_axon = 100 * cm[0][0] / (cm[0][0] + cm[0][1])
                recall_axon = "{:>.2f}".format(recall_axon) + "%"
            else:
                recall_axon = "nan"
            if cm[1][1] + cm[1][0] != 0:
                recall_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[1][0])
                recall_dendrite = "{:>.2f}".format(recall_dendrite) + "%"
            else:
                recall_dendrite = "nan"
            if cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] != 0:
                accuracy = 100 * (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
                accuracy = "{:>.2f}".format(accuracy) + "%"
            else:
                accuracy = "nan"

            print("==========================================================")
            print("Neuron Set: ", i)
            print("model1: " + lst_of_df[index1][0])
            print("model2: " + lst_of_df[index2][0])
            print("lost points: ", lost_points)
            print('==========================================================')
            print('| Real \\ Pred |   Axon   | Dendrite |')
            print('----------------------------------------------------------')
            print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Axon", cm[0][0], cm[0][1]))
            print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Dendrite", cm[1][0], cm[1][1]))
            print("==========================================================")
            print('|             | Precision |  Recall   |')
            print('----------------------------------------------------------')
            print("|{:^13s}|{:^11s}|{:^11s}|".format("Axon", precision_axon, recall_axon))
            print("|{:^13s}|{:^11s}|{:^11s}|".format("Dendrite", precision_dendrite, recall_dendrite))
            print("Accuracy: {:<8s}".format(accuracy))
            print("==========================================================\n")

            # output csv
            output_csv[num * 11 + 2] += [i, "", "", ""]
            output_csv[num * 11 + 3] += ["Accuracy: ", accuracy, "", ""]
            output_csv[num * 11 + 4] += ["", "Axon", "Dendrite", ""]
            output_csv[num * 11 + 5] += ["Axon", str(cm[0][0]), str(cm[0][1]), ""]
            output_csv[num * 11 + 6] += ["Dendrite", str(cm[1][0]), str(cm[1][1]), ""]
            output_csv[num * 11 + 7] += ["", "Precision", "Recall", ""]
            output_csv[num * 11 + 8] += ["Axon", precision_axon, recall_axon, ""]
            output_csv[num * 11 + 9] += ["Dendrite", precision_dendrite, recall_dendrite, ""]

    # save csv
    with open('./data/nrn_result/hyper_result_nodes.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(output_csv)):
            writer.writerow(output_csv[i])

    output_csv = []
    print("Level: Neurons")
    df = lst_of_df[0][1]
    df.sort_values(by=['nrn', 'ID'])
    id_check = df.loc[:, ['nrn', 'ID', 'NC', 'type_pre']]
    # threshold combination process
    print("Hyper Relabel: Combination Process")
    sep_models = {}
    for num in range(len(comparison)):
        index1 = comparison[num][0]
        index2 = comparison[num][1]
        name = "model" + "_" + str(index1) + "_" + str(index2) + "_lst"
        sep_models[name] = hyper_process_neuron(models[model_lst[index1]], models[model_lst[index2]], threshold,
                                                lst_of_df[index1][0], lst_of_df[index2][0], point_s, reduce_th)

        # output csv
        output_csv += [[] for i in range(11)]
        output_csv[num*11] += ["Model1: ", lst_of_df[index1][0]]
        output_csv[num*11+1] += ["Model2: ", lst_of_df[index2][0]]
        output_csv[num*11+10] += [""]

    # relabel process
    print("Hyper Relabel: Post-Relabel Process")
    keys = list(sep_models.keys())
    for key in keys:
        sep_models[key] = relabel_process(sep_models[key], point_s)
        sep_models[key] = pd.DataFrame(sep_models[key], columns=['nrn', 'ID', 'PARENT_ID', 'NC', 'type_post'])
        sep_models[key] = sep_models[key].loc[:, ['nrn', 'ID', 'type_post']]

    # check result Nodes
    print("Hyper Relabel: Print Results")
    df = lst_of_df[0][1]
    df.sort_values(by=['nrn', 'ID'])
    id_check = df.loc[:, ['nrn', 'ID', 'NC', 'type_pre']]
    for num in range(len(comparison)):
        index1 = comparison[num][0]
        index2 = comparison[num][1]
        name = "model" + "_" + str(index1) + "_" + str(index2) + "_lst"
        res = pd.merge(id_check, sep_models[name], how='left', on=['nrn', 'ID'])
        res = res.values.tolist()
        cm = [[0, 0], [0, 0]]
        lost_points = 0
        for i in range(len(res)):
            if res[i][2] == 0:
                if res[i][3] == 2:
                    if res[i][4] == 2:
                        cm[0][0] += 1
                    elif res[i][4] == 3:
                        cm[0][1] += 1
                    else:
                        lost_points += 1
                elif res[i][3] == 3:
                    if res[i][4] == 2:
                        cm[1][0] += 1
                    elif res[i][4] == 3:
                        cm[1][1] += 1
                    else:
                        lost_points += 1

        if cm[0][0] + cm[1][0] != 0:
            precision_axon = 100 * cm[0][0] / (cm[0][0] + cm[1][0])
            precision_axon = "{:>.2f}".format(precision_axon) + "%"
        else:
            precision_axon = "nan"
        if cm[1][1] + cm[0][1] != 0:
            precision_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[0][1])
            precision_dendrite = "{:>.2f}".format(precision_dendrite) + "%"
        else:
            precision_dendrite = "nan"
        if cm[0][0] + cm[0][1] != 0:
            recall_axon = 100 * cm[0][0] / (cm[0][0] + cm[0][1])
            recall_axon = "{:>.2f}".format(recall_axon) + "%"
        else:
            recall_axon = "nan"
        if cm[1][1] + cm[1][0] != 0:
            recall_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[1][0])
            recall_dendrite = "{:>.2f}".format(recall_dendrite) + "%"
        else:
            recall_dendrite = "nan"
        accuracy = 100 * (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])

        print("==========================================================")
        print("Neuron Set: total")
        print("model1: " + lst_of_df[index1][0])
        print("model2: " + lst_of_df[index2][0])
        print("lost points: ", lost_points)
        print('==========================================================')
        print('| Real \\ Pred |   Axon   | Dendrite |')
        print('----------------------------------------------------------')
        print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Axon", cm[0][0], cm[0][1]))
        print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Dendrite", cm[1][0], cm[1][1]))
        print("==========================================================")
        print('==========================================================')
        print('|             | Precision |  Recall   |')
        print('----------------------------------------------------------')
        print("|{:^13s}|{:^11s}|{:^11s}|".format("Axon", precision_axon, recall_axon))
        print("|{:^13s}|{:^11s}|{:^11s}|".format("Dendrite", precision_dendrite, recall_dendrite))
        print("Accuracy: {:<.2f}%".format(accuracy))
        print("==========================================================\n")

        for i in n_set:
            cm = [[0, 0], [0, 0]]
            lost_points = 0
            for j in range(len(res)):
                if res[j][0] in neuron_set[i]:
                    if res[j][2] == 0:
                        if res[j][3] == 2:
                            if res[j][4] == 2:
                                cm[0][0] += 1
                            elif res[j][4] == 3:
                                cm[0][1] += 1
                            else:
                                lost_points += 1
                        elif res[j][3] == 3:
                            if res[j][4] == 2:
                                cm[1][0] += 1
                            elif res[j][4] == 3:
                                cm[1][1] += 1
                            else:
                                lost_points += 1
            if cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] == 0:
                continue

            if cm[0][0] + cm[1][0] != 0:
                precision_axon = 100 * cm[0][0] / (cm[0][0] + cm[1][0])
                precision_axon = "{:>.2f}".format(precision_axon) + "%"
            else:
                precision_axon = "nan"
            if cm[1][1] + cm[0][1] != 0:
                precision_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[0][1])
                precision_dendrite = "{:>.2f}".format(precision_dendrite) + "%"
            else:
                precision_dendrite = "nan"
            if cm[0][0] + cm[0][1] != 0:
                recall_axon = 100 * cm[0][0] / (cm[0][0] + cm[0][1])
                recall_axon = "{:>.2f}".format(recall_axon) + "%"
            else:
                recall_axon = "nan"
            if cm[1][1] + cm[1][0] != 0:
                recall_dendrite = 100 * cm[1][1] / (cm[1][1] + cm[1][0])
                recall_dendrite = "{:>.2f}".format(recall_dendrite) + "%"
            else:
                recall_dendrite = "nan"
            if cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] != 0:
                accuracy = 100 * (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
                accuracy = "{:>.2f}".format(accuracy) + "%"
            else:
                accuracy = "nan"

            print("==========================================================")
            print("Neuron Set: ", i)
            print("model1: " + lst_of_df[index1][0])
            print("model2: " + lst_of_df[index2][0])
            print("lost points: ", lost_points)
            print('==========================================================')
            print('| Real \\ Pred |   Axon   | Dendrite |')
            print('----------------------------------------------------------')
            print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Axon", cm[0][0], cm[0][1]))
            print("|  {:<9s}  |{:^10d}|{:^10d}|".format("Dendrite", cm[1][0], cm[1][1]))
            print("==========================================================")
            print('|             | Precision |  Recall   |')
            print('----------------------------------------------------------')
            print("|{:^13s}|{:^11s}|{:^11s}|".format("Axon", precision_axon, recall_axon))
            print("|{:^13s}|{:^11s}|{:^11s}|".format("Dendrite", precision_dendrite, recall_dendrite))
            print("Accuracy: {:<8s}%".format(accuracy))
            print("==========================================================\n")

            # output csv
            output_csv[num * 11 + 2] += [i, "", "", ""]
            output_csv[num * 11 + 3] += ["Accuracy: ", accuracy, "", ""]
            output_csv[num * 11 + 4] += ["", "Axon", "Dendrite", ""]
            output_csv[num * 11 + 5] += ["Axon", str(cm[0][0]), str(cm[0][1]), ""]
            output_csv[num * 11 + 6] += ["Dendrite", str(cm[1][0]), str(cm[1][1]), ""]
            output_csv[num * 11 + 7] += ["", "Precision", "Recall", ""]
            output_csv[num * 11 + 8] += ["Axon", precision_axon, recall_axon, ""]
            output_csv[num * 11 + 9] += ["Dendrite", precision_dendrite, recall_dendrite, ""]

    # save csv
    with open('./data/nrn_result/hyper_result_neurons.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(output_csv)):
            writer.writerow(output_csv[i])

    return


########################################################################################################################
def data_dist(d_lst, length):
    set = []

    random_set = random.sample(range(length), length)
    set.append(random_set[0: int(length / sum(d_lst))])
    for i in range(1, sum(d_lst)):
        set.append(random_set[int(i*length / sum(d_lst)): int((i+1)*length / sum(d_lst))])

    return set


########################################################################################################################
def list_to_np(data, label, chosen_set):
    input_lst = []
    label_lst = []
    for i in chosen_set:
        input_lst.append(data[i])
        label_lst.append(label[i])
    input_lst = np.array(input_lst)
    label_lst = np.array(label_lst)

    return input_lst, label_lst


########################################################################################################################
def run_NN(file_name, data):
    """
        Reconstruct NN and use the trained weights(and bias) to do the prediction

        :param file_name:
            type: str
            the archive path for the target NN

        :param data:
            type: np.array
            shape[0]: number of test nodes
            shape[1]: length of features

        :return:
            type: np.array
            shape[0]: number of test nodes
            shape[1]: prediction --> probability
    """

    def activation_select(input_str):
        if input_str == 'sigmoid':
            return tf.nn.sigmoid
        elif input_str == 'relu':
            return tf.nn.relu
        elif input_str == 'selu':
            return tf.nn.selu

    with open(file_name, 'rb') as file:
        NN_info = pickle.load(file)

    # modify the type
    data = data.astype("float32")

    activation = activation_select(NN_info['NN_list'][0])
    L1 = activation(tf.matmul(data, NN_info['W1']) + NN_info['b1'])
    c_v = locals()
    for i in range(len(NN_info['NN_list']) - 1):
        activation = activation_select(NN_info['NN_list'][i+1])
        c_v['L' + str(i + 2)] = activation(tf.matmul(c_v['L' + str(i + 1)],
                                                     NN_info['W' + str(i + 2)]) + NN_info['b' + str(i + 2)])
    c_v['output'] = tf.matmul(c_v['L' + str(len(NN_info['NN_list']))], NN_info['W' + str(len(NN_info['NN_list'])+1)])+NN_info['b' + str(len(NN_info['NN_list'])+1)]

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        prediction_list = sess.run(c_v['output'])
        output = np.zeros([data.shape[0], 1])
        for i in range(data.shape[0]):
            temp = prediction_list[i][0] - prediction_list[i][1]
            if temp > 10:
                output[i][0] = 0
            elif temp < -10:
                output[i][0] = 1
            else:
                output[i][0] = 1 / (1 + math.exp(temp))
    output = output.reshape((output.shape[0],))

    return output


########################################################################################################################
def pyramid(df, feature_list, num_layer, th_layer, for_train=False, empty_number=-1):
    """
        Give

        :param df:
            type: DataFrame
            only permit df of one neuron list (level tree)
        :param feature_list:
            type: list
            the features. For example: ['s', 'norm_s']
        :param num_layer:
            type: int
            the number of pyramid layer you want --> 2^num-1 nodes
        :param th_layer:
            type: int
            if the number of layers is less than this parameter, then it will be ignored.
        :param for_train:
            type: bool
            the pyramid is for training or not, i.e. with label or not

        :return:
            type: DataFrame
            for example, you give
                feature_list = ['s', 'norm_s']
                num_layer = 2
                th_layer = 1

                the output --> df["nrn", "ID", "T", "s_1", "norm_s_1", "s_2", "norm_s_2", "s_3", "norm_s_3"]
                "T" --> 0:dendrite, 1:axon
    """

    def index_searchv2(lst, value):
        index_p = -1
        for i in range(len(lst)):
            if lst[i][1] == value:
                index_p = i
                break

        return index_p

    def children_searchv2(lst, value):
        index_c = []
        for i in range(len(lst)):
            if lst[i][2] == value:
                index_c.append(i)

        return index_c

    def pyramid_structure(lst, n, layer, chosen_para, e_num):
        pyramid = []
        for i in range(layer):
            if i == 0:
                index = index_searchv2(lst, n)
                temp = [lst[index][l] for l in chosen_para]
                pyramid.append(temp)
                del temp
                anc = [n]
                continue

            temp_anc = []
            for j in anc:
                if j == -1:
                    temp_anc.append(e_num)
                    temp_anc.append(e_num)
                    pyramid.append([e_num for l in chosen_para])
                    pyramid.append([e_num for l in chosen_para])
                else:
                    index = children_searchv2(lst, j)
                    if len(index) >= 3:
                        change = True
                        while change:
                            change = False
                            for k in range(1, len(index)):
                                if lst[index[k - 1]][4] < lst[index[k]][4]:
                                    temp = index[k - 1]
                                    index[k - 1] = index[k]
                                    index[k] = temp
                                    change = True
                        for k in range(2):
                            temp_anc.append(lst[index[k]][1])
                        for k in range(2):
                            temp = [lst[index[k]][l] for l in chosen_para]
                            pyramid.append(temp)
                    if len(index) == 2:
                        if lst[index[0]][4] < lst[index[1]][4]:
                            temp = index[0]
                            index[0] = index[1]
                            index[1] = temp
                        for k in range(2):
                            temp_anc.append(lst[index[k]][1])
                        for k in range(2):
                            temp = [lst[index[k]][l] for l in chosen_para]
                            pyramid.append(temp)
                    if len(index) == 0:
                        temp_anc.append(e_num)
                        temp_anc.append(e_num)
                        pyramid.append([e_num for l in chosen_para])
                        pyramid.append([e_num for l in chosen_para])
            anc = temp_anc
        temp = pyramid[0]
        for i in range(1, len(pyramid)):
            for j in range(len(pyramid[i])):
                temp.append(pyramid[i][j])
        pyramid = temp
        del temp

        return pyramid

    def list_to_train_data(lst, num_layer, threshold_layer, para_list, file_name, chosen_para, train, e_num):
        # list --> train data
        # check the layer of pyramid
        nodes = []
        for i in range(len(lst)):
            if lst[i][5] == 2 or lst[i][5] == 3:
                nodes.append(i)
                continue
        result = []
        for i in nodes:
            fulfill = False
            index = children_searchv2(lst, lst[i][1])
            if len(index) == 0:
                count = 1
            else:
                count = 2
            while not fulfill:
                if count >= threshold_layer:
                    result.append(i)
                    break

                temp = []
                for j in index:
                    if lst[j][3] != 0:
                        for k in children_searchv2(lst, lst[j][1]):
                            temp.append(k)
                index = temp
                del temp

                if len(index) > 0:
                    count += 1
                else:
                    fulfill = True

        if len(result) == 0:
            return pd.DataFrame([])
        # create train data
        train_data = []
        label_data = []
        nrn = []
        for i in result:
            nrn.append([file_name, lst[i][1]])
            pyramid = pyramid_structure(lst, lst[i][1], num_layer, para_list, e_num)
            train_data.append(pyramid)
            if lst[i][5] == 2:
                label_data.append(1)  # Axon
            if lst[i][5] == 3:
                label_data.append(0)  # Dendirte
        nrn_t = list(map(list, zip(*nrn)))
        res_dict = {
            "nrn": nrn_t[0],
            "ID": nrn_t[1]
        }
        if train:
            res_dict["label"] = label_data

        train_data_t = list(map(list, zip(*train_data)))
        temp = 0
        col = []
        for i in range(len(train_data_t)):
            if i % len(chosen_para) == 0:
                temp += 1
            col.append(chosen_para[i % len(chosen_para)] + "_" + str(temp))
        for i in range(len(train_data_t)):
            res_dict[col[i]] = train_data_t[i]

        res_df = pd.DataFrame(res_dict)

        return res_df

    def create_train_data(lst, num_layer, threshold_layer, chosen_para, train, e_num):
        # change form of chosen_para
        para_list = [6 + i for i in range(len(chosen_para))]

        # check the list of neuron files
        file_names = []
        for i in range(len(lst)):
            if lst[i][0] not in file_names:
                file_names.append(lst[i][0])

        # create the right form of training data
        res_df = pd.DataFrame()
        start = 0

        # sort the neuron

        # pick one neuron
        for i in file_names:
            temp_file_name = i
            temp_lst = []
            touch = False
            for j in range(start, len(lst)):
                if lst[j][0] == i:
                    touch = True
                    temp_lst.append(lst[j])
                if touch:
                    if lst[j][0] != i:
                        # start = j
                        break

            temp_df = list_to_train_data(temp_lst, num_layer, threshold_layer,
                                         para_list, temp_file_name, chosen_para, train, e_num)
            res_df = pd.concat([res_df, temp_df], ignore_index=True)

        return res_df

    if for_train:
        df_p = df.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'dp_level', 'type_pre'] + feature_list]
        df_lst = df_p.values.tolist()
        pyra = create_train_data(df_lst, num_layer, th_layer, feature_list, for_train, empty_number)
    else:
        df_p = df.loc[:, ['nrn', 'ID', 'PARENT_ID', 'NC', 'dp_level'] + feature_list]
        fk_lst = [2 for i in range(df_p.shape[0])]
        df_p.insert(loc=5, column='type_pre', value=fk_lst)
        df_lst = df_p.values.tolist()
        pyra = create_train_data(df_lst, num_layer, th_layer, feature_list, for_train, empty_number)

    return pyra


########################################################################################################################
def update_parent_col(df, df_dis, tree_node_dict, child_col, parent_col, select_node=["root", "fork", "leaf"]):
    # Select nodes
    if set(select_node).issubset(set(["root", "fork", "leaf"])):
        tn_lst = []
        for i in select_node:
            tn_lst += tree_node_dict[i]
        df = df.loc[df[child_col].isin(tn_lst)]
    else:
        sys.exit("\n select_node should be in this range ['root', 'fork', 'leaf']! Check update_parent_col().")
    # Merge df_dic ancestor to df
    _info = df_dis.loc[:, ['descendant', "ancestor"]]
    _info = _info.rename(columns={'descendant': child_col})
    df = pd.merge(df, _info, how="left", on=child_col)
    # Update root's parent
    df[parent_col] = np.where(df[child_col] == tree_node_dict["root"][0], -1, df["ancestor"])
    # Clean up
    df = df.drop(["ancestor"], 1)
    df[[parent_col]] = df[[parent_col]].astype('int')

    return df






########################################################################################################################

