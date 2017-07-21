import django, os, sys, time, resource, re, gc, shutil
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.db.models import Count, Avg
from multiprocess import Pool
from functools import partial
import numpy as np
from functools import partial
from scipy.sparse import coo_matrix, csr_matrix, find, tril
import networkx as nx

sys.path.append('/home/galm/software/tmv/BasicBrowser/')

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *
from tmv_app.models import *
qid = 1457

df = pd.read_excel(
    "1457_fastgreedy_filtered_weight.xlsx",
    header = None,
    names = ["id","cluster","wosid","title"]
)

q = Query.objects.get(pk=1457)

network, created = Network.objects.get_or_create(
    title="fast_greedy_weighted_2017_07_19",
    type="fast_greedy_weighted",
    query=q
)

network.save()

for index, row in df.iterrows():
    try:
        netprop, created = NetworkProperties.objects.get_or_create(
            doc = Doc.objects.get(pk=row["wosid"]),
            network = network
        )
        netprop.value = int(row["cluster"])
        netprop.save()
    except:
        print(row)
