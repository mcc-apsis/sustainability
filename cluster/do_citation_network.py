import numpy as np

from scipy.sparse import csr_matrix, find, tril
import csv
import igraph


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

m = load_sparse_csr("1457_citations.npz")

#subg = list(range(10000))
#m = m.tocsr()[subg, :].tocsc()[:, subg]

nodes = m.shape[0]

print(nodes)

with open("1457_nodelabels.txt") as f:
    c = csv.reader(f,delimiter='\t')
    ids = [r[1] for r in c]


g = igraph.Graph()
g.add_vertices(nodes)

mat = find(m)
edge_n = len(mat[0])

g.add_edges(zip(mat[0],mat[1]))

igraph.summary(g)

g.vs["wosid"] = ids

giant = g.clusters().giant().simplify()

igraph.summary(giant)

igraph.plot(giant,"connected_network.pdf")

igraph.plot(g,"network.pdf")


dendogram = giant.community_fastgreedy()

clusters = dendogram.as_clustering()

membership = clusters.membership

with open("output","w") as w:
    writer = csv.writer(w)
    for i in range(len(membership)):
        name = giant.vs["wosid"][i]
        mem = membership[i]
        writer.writerow([name,mem])

    

