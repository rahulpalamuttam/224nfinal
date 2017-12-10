import fileinput
import snap
import random
import numpy as np
from numpy.linalg import eig
import scipy
from scipy import sparse
import scipy.sparse.linalg as la
graph = snap.LoadEdgeList(snap.PUNGraph, "/Users/rahulpalamuttam/cs224_final/data/BlogCatalog-dataset/data/edges.edges", 0, 1)

labels = []
# for line in fileinput.input("polblogs-labels.txt"):
#     if line=="0\n":
#         labels.append(-1)
#     else:
#         labels.append(1)

num_nodes = graph.GetNodes()
print "GRAPH NUM NODES : ", graph.GetNodes()

A = np.zeros((num_nodes, num_nodes))

for e in graph.Edges():
    src, dst = e.GetSrcNId() - 1, e.GetDstNId() - 1
    A[src][dst] = 1.0
    A[dst][src] = 1.0

A = sparse.csc_matrix(A)

D = np.zeros((num_nodes, num_nodes))
D_h = np.zeros((num_nodes, num_nodes))
for node in graph.Nodes():
    id = node.GetId() - 1
    D[id][id] = node.GetDeg() * 1.0
    D_h[id][id] = (node.GetDeg() * 1.0) ** -0.5

D = sparse.csc_matrix(D)
D_h = sparse.csc_matrix(D)
print D
print D_h

L = D - A
print L

L_b = np.dot(D_h,L).dot(D_h)
print "NORMALIZED LAPLACIAN : \n", L_b

L_b = sparse.csc_matrix(L_b)

eigval, eigvec = la.eigs(L_b, k=260)
print eigval
print eigvec

idx = eigval.argsort()[::-1]
print "SORT INDEX ", idx
eigval = eigval[idx][0:130]
eigvec = np.real(eigvec[:,idx][:,0:130])

print eigvec.shape
filename = "eigenvec_130_embeddings.emd"
file = open(filename, 'w+')

for nid in range(0, len(eigvec)):
#    print nid
#    print vec
    vec = eigvec[nid]
    stremb = " ".join([str(i) for i in vec])
    file.write("%s %s\n" % (nid + 1, stremb))

file.close()
    
    
