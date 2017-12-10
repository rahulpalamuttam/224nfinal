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
import argparse
import pickle as pkl

# Prints stats for the graph including generate T-sne
def print_stats(G, embeddings):
  data = embeddings[:, 1:-1]
  labels = embeddings[:,-1]

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
  tsne_results = tsne.fit_transform(data)
  scatter(tsne_results, labels)
  scatter([], labels)
  #snap.DrawGViz(G, snap.gvlSfdp, "graph.ps", "graph", True)

# Plot T-sne
def scatter(x, colors):
  plt.scatter(x[:, 0].flatten(), x[:,1].flatten(), c=labels)
#  sc = plt.scatter(np.array(x[:,0].tolist()), np.array(x[:,1].tolist()), c=np.array(colors.tolist()))
  plt.show()

# Appends the circle ids to the end of each nodes's feature vector
def parse_blog_circles(fn, nodes):
  node_to_circle = defaultdict(int)
  with open(fn) as f:
    for line in f:
      nid, cid = line.strip().split()
      cid = str(int(cid) - 1)
      if int(nid) not in node_to_circle:
        node_to_circle[int(nid)] = int(cid)

  # Add a label for no circle
  # for i in nodes:
  #   if int(i) not in node_to_circle: node_to_circle[int(i)] = [0]

  return node_to_circle

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

# ================================================================================ 
parser = argparse.ArgumentParser(description="Run node2vec.")

parser.add_argument('--input', nargs='?', default='BlogCatalog-dataset/data/')
args = parser.parse_args()

directory = args.input

x = open('ind.cora.allx')
y = open('ind.cora.ally')

x = pkl.load(x).todense()
y = pkl.load(y)

labels = []
for i in range(0, y.shape[0]):
  for j in range(0, y.shape[1]):
    if y[i][j] == 1:
      labels.append(j)
      break

labels = np.array(labels)

graph = snap.LoadEdgeList(snap.PUNGraph, directory + 'edges.edges', 0, 1)

edges_fn = directory + 'eigenvec_130_embeddings.emd'
circles_fn = directory + 'circles.circles'

x = np.zeros(x.shape) + x
y = np.zeros(y.shape) + y
print(x)
print(y)
print x.shape
print type(x)
print y.shape
print type(y)
x = np.append(x, np.zeros((y.shape[0], 1)), 1)
# TODO: handle multiple circle assignments?
for i in xrange(y.shape[0]):
  x[i][-1] = labels[i]

print_stats(graph, x)
