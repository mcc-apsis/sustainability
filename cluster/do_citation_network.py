import numpy as np

from scipy.sparse import csr_matrix, find, tril
import csv
import igraph


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

m = load_sparse_csr("1457_citations_lutz.npz")

#subg = list(range(10000))
#m = m.tocsr()[subg, :].tocsc()[:, subg]

nodes = m.shape[0]

print(nodes)

ids = []
titles = []

with open("1457_nodelabels_lutz.txt") as f:
    c = csv.reader(f,delimiter='\t')
    for r in c:
        i = r[1]
        t = r[0]
        ids.append(i)
        titles.append(t)
        


g = igraph.Graph()
g.add_vertices(nodes)

mat = find(m)
edge_n = len(mat[0])

g.add_edges(zip(mat[0],mat[1]))

igraph.summary(g)

g.vs["wosid"] = ids
g.vs["title"] = titles

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
        ti = giant.vs["title"][i]
        mem = membership[i]
        writer.writerow([name,ti,mem])

    

