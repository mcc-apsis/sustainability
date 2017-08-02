import numpy as np

from scipy.sparse import csr_matrix, find, tril
import csv, sys
import igraph


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


g = igraph.Graph()
g.add_vertices(nodes)

mat = find(m)
edge_n = len(mat[0])

g.add_edges(zip(mat[0],mat[1]))

g.es["weight"] = mat[2]

igraph.summary(g)

g.vs["wosid"] = ids
g.vs["title"] = titles

giant = g.clusters().giant().simplify()

igraph.summary(giant)



visual_style = {}

visual_style["edge_width"] = 0.001
visual_style["vertex_size"] = 0.1
visual_style["edge_color"] = "rgba(1,1,1,0.1)"

igraph.plot(giant,"connected_network_bc.pdf", **visual_style)

#igraph.plot(g,"network.pdf", **visual_style)


dendogram = giant.community_fastgreedy(
    weights = giant.es["weight"]
)

clusters = dendogram.as_clustering()

cn = 0

visual_style["vertex_label"] = ["" for x in clusters.membership]

membership = clusters.membership

for c in clusters:
    cn+=1
    degrees = [giant.degree(x) for x in c]
    i = degrees.index(max(degrees))
    visual_style["vertex_label"][i] = membership[i]


#visual_style["vertex_label"] = membership

#igraph.plot(dendogram,"dend_network.pdf", mark_groups=True, **visual_style)

igraph.plot(clusters,"clust_network_bc.pdf", mark_groups=True, **visual_style)

with open("output_bc.csv","w") as w:
    writer = csv.writer(w,delimiter='\t')
    for i in range(len(membership)):
        name = giant.vs["wosid"][i]
        ti = giant.vs["title"][i]
        mem = membership[i]
        writer.writerow([name,ti,mem])
