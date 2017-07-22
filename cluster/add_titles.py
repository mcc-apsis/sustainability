import numpy as np

from scipy.sparse import csr_matrix, find, tril
import csv
import igraph

with open("1457_nodelabels.txt") as f:
    c = csv.reader(f,delimiter='\t')
    mydict = {rows[1].strip():rows[0] for rows in c}



with open("results.csv","w") as res:
    writer = csv.writer(res)
    with open("output") as o:
        for l in o:           
            fields = [x.strip() for x in l.split(',')]
            try:
                t = mydict[fields[0]]
            except:
                t = None
            fields.append(t)
            writer.writerow(fields)
    
