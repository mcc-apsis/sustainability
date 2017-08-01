import django, os, sys, time, resource, re, gc, shutil
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.db.models import Count, Avg, F
from multiprocess import Pool
from functools import partial
import numpy as np
from functools import partial
from scipy.sparse import coo_matrix, csr_matrix, find, tril
import networkx as nx
import random

sys.path.append('/home/galm/software/tmv/BasicBrowser/')

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *
from tmv_app.models import *
qid = 1457
q = Query.objects.get(pk=qid)
docs = Doc.objects.filter(query=q)

health=docs.filter(wc__oecd='Medical and Health Sciences')
health.count()

gbgs = [
    'develop','agricultur','energi',
    'product','forest','water','land',
    'citi','public','use','urban','futur',
    'growth','transport','tourism','harvest',
    'food','crop','livelihood'
]
gbg_1s = [
    'environment','ecolog','social','agricultur'
]
good_health = health.filter(
    docbigram__bigram__pos=1,
    docbigram__bigram__stem2__in=gbgs
) | health.filter(
    docbigram__bigram__pos=-1,
    docbigram__bigram__stem2__in=gbg_1s
)
bad_health = health.exclude(
    docbigram__bigram__pos=1,
    docbigram__bigram__stem2__in=gbgs
).exclude(
    docbigram__bigram__pos=-1,
    docbigram__bigram__stem2__in=gbg_1s
)
# print(good_health.distinct().count())
# print(bad_health.distinct().count())


bdocs = docs.filter(wc__oecd_fos_text='Economics and business')

badbgs = [
    'cooper','competit','equilibrium',
    'collus','advantag','profit','tacit',
    'change','improv','outcom','success',
    'allianc','tqm','brand'
]
badbg_1s = [
    'collus','cartel','roi','leverag','price'
]
bad_business = bdocs.filter(
    docbigram__bigram__pos=1,
    docbigram__bigram__stem2__in=badbgs,

) | bdocs.filter(
    docbigram__bigram__pos=-1,
    docbigram__bigram__stem2__in=badbg_1s
)
good_business = bdocs.exclude(
    docbigram__bigram__pos=1,
    docbigram__bigram__stem2__in=badbgs
).exclude(
    docbigram__bigram__pos=-1,
    docbigram__bigram__stem2__in=badbg_1s
)
# print(good_business.distinct().count())
# print(bad_business.distinct().count())

# t = Tag(
#     query=q,
#     title="good_business_5"
# )
# t.save()
# for d in good_business:
#     d.tag.add(t)
#
# t = Tag(
#     query=q,
#     title="bad_business_5"
# )
# t.save()
# for d in bad_business:
#     d.tag.add(t)


bad_ids = bad_business | bad_health
bad_ids = list(bad_ids.distinct().values_list('UT',flat=True))

good_docs = docs.exclude(UT__in=bad_ids)
bad_docs = docs.filter(UT__in=bad_ids)

# t = Tag(
#     query=q,
#     title="good_docs_3"
# )
# t.save()
for d in good_docs:
    # d.tag.add(t)
    d.relevant=True
    d.save()

# t = Tag(
#     query=q,
#     title="bad_docs_3"
# )
# t.save()
for d in bad_docs:
    # d.tag.add(t)
    d.relevant=False
    d.save()
