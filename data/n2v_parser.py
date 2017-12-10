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

# Prints stats for the graph including generate T-sne
def print_stats(G, embeddings):
  print 'Printing graph stats'
  num_nodes = G.GetNodes()
  num_edges = G.GetEdges()
  print 'Number of nodes: ' + str(num_nodes)
  print 'Number of edges: ' + str(num_edges)

  data = embeddings[:, 1:-1]
  labels = embeddings[:,-1]

  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
  tsne_results = tsne.fit_transform(data)
  scatter(tsne_results, labels)

  #snap.DrawGViz(G, snap.gvlSfdp, "graph.ps", "graph", True)

# Plot T-sne
def scatter(x, colors):
  sc = plt.scatter(x[:,0], x[:,1], c=colors)
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
graph = snap.LoadEdgeList(snap.PUNGraph, directory + 'edges.edges', 0, 1)

edges_fn = directory + 'eigenvec_130_embeddings.emd'
circles_fn = directory + 'circles.circles'


# embeddings has the node_id as first column, circle_id in the last column, and the feature
# vector inbetween
embeddings = np.loadtxt(edges_fn)
eshape = embeddings.shape
nodes = set(embeddings[:,0])
#embeddings = embeddings[:,1:]

mapping = parse_blog_circles(circles_fn, nodes)

embeddings = np.append(embeddings, np.zeros((eshape[0], 1)), 1)
# TODO: handle multiple circle assignments?
for i in xrange(eshape[0]):
  embeddings[i][-1] = mapping[int(embeddings[i][0])]

print_stats(graph, embeddings)
