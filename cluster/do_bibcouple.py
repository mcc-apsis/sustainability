import numpy as np

from scipy.sparse import csr_matrix, find, tril
import csv, sys
#import igraph


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

m = load_sparse_csr("1457_bc_lutz_3.npz")

#subg = list(range(10000))
#m = m.tocsr()[subg, :].tocsc()[:, subg]

nodes = m.shape[0]

print(nodes)

ids = []
titles = []

with open("1457_nodelabels.txt") as f:
    c = csv.reader(f,delimiter='\t')
    for row in c:
        ids.append(row[1])
        titles.append(row[0])

sys.exit()

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

igraph.plot(giant,"connected_network_bc.pdf")

igraph.plot(g,"network_bc.pdf")


dendogram = giant.community_fastgreedy()

clusters = dendogram.as_clustering()

membership = clusters.membership

with open("output_bc.csv","w") as w:
    writer = csv.writer(w, delimiter='\t')
    for i in range(len(membership)):
        name = giant.vs["wosid"][i]
        title = giant.vs["title"][i]
        mem = membership[i]
        writer.writerow([name,title,mem])
