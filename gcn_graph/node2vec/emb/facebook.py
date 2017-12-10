import sys
sys.path.insert(0, '../../')
import snap
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import cPickle
import random
from random import shuffle
import scipy

# Appends the circle ids to the end of each nodes's feature vector
def parse_circles(fn, nodes):
  node_to_circle = defaultdict(int)
  with open(fn) as f:
    for line in f:
      split = line.strip().split()
      for node in split[1:]:
        nid = int(node)
        # Ignore nodes that are listed in circles file but not in the edges file
        if nid not in nodes: continue
        if nid not in node_to_circle:
          node_to_circle[nid] = [int(split[0][6:])]
        else:
          node_to_circle[nid].append(int(split[0][6:]))

  # Add a label for no circle
  for i in nodes:
    if int(i) not in node_to_circle: node_to_circle[int(i)] = [9]

  return node_to_circle


graph = snap.LoadEdgeList(snap.PUNGraph, '../graph/107.edges', 0, 1)
nodes = [j.GetId() for j in graph.Nodes()]
node_map = {}
for num, n in enumerate(nodes):
  node_map[n] = num

max_nodes = max(nodes)
print(max_nodes)

dict_nodes = {}
adjx = np.zeros((len(nodes), len(nodes)))
for node in graph.Nodes():
    j = node.GetId()
    j_map = node_map[j]
    if j_map not in dict_nodes:
        dict_nodes[j_map] = []
    for Id in node.GetOutEdges():
        dict_nodes[j_map].append(node_map[Id])
        adjx[j_map][node_map[Id]] = 1.0
circles_fn = '../graph/107.circles'

# embeddings has the node_id as first column, circle_id in the last column, and the feature
# vector inbetween
mapping = parse_circles(circles_fn, nodes)
classes = []

nodes = scipy.sparse.csr_matrix(adjx)

for k in mapping:
    one_hot = np.zeros(10)
    one_hot[mapping[k]] = 1
    classes.append(one_hot)
classes = np.array(classes)

with open('ind.facebook.graph','wb') as fp:
    cPickle.dump(dict_nodes, fp)

with open('ind.facebook.allx', 'wb') as fp:
    cPickle.dump(nodes, fp)

with open('ind.facebook.x', 'wb') as fp:
    cPickle.dump(nodes[0:800],fp)

with open('ind.facebook.tx', 'wb') as fp:
    cPickle.dump(nodes[800:],fp)

with open('ind.facebook.ally', 'wb') as fp:
    cPickle.dump(classes,fp)

with open('ind.facebook.y', 'wb') as fp:
    cPickle.dump(classes[0:800],fp)

with open('ind.facebook.ty', 'wb') as fp:
    cPickle.dump(classes[800:],fp)

reorder = np.arange(0, len(classes[800:]))
print reorder
random.shuffle(reorder)
reorder = np.array(reorder)
print reorder

with open('ind.facebook.test.index', 'wb') as fp:
  for k in reorder:
    fp.write('%s\n' % k)

print dict_nodes
print graph.GetNodes()
