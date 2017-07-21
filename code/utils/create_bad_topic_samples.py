import pickle, string, numpy, getopt, sys, random, time, re, pprint, gc, resource
import pandas as pd
import onlineldavb, scrapeWoS, gensim, nltk, subprocess, psycopg2, math
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from multiprocess import Pool
import django
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from time import time
from django.utils import timezone
from scipy.sparse import csr_matrix, find
import numpy as np


sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db as db
from tmv_app.models import *
from scoping.models import *

btopics = [25674,25741,25771,25713,25738,25775,25561,25772]
btopics = []

q = Query.objects.get(pk=1457)

for t in btopics:
    topic = Topic.objects.get(pk=t)
    tdocs = Doc.objects.filter(
        doctopic__topic=t,
        doctopic__score__gt=0.01
    )
    tag = Tag(title=topic.title)
    tag.query=q
    tag.save()
    for d in tdocs:
        d.tag.add(tag)
    print(tdocs.count())

meds = WC.objects.filter(
    oecd__icontains="Medic",
    doc__query=1457
).distinct('text')

meds = []
for m in meds:
    mdocs = Doc.objects.filter(
        query=1457,
        wc=m
    )
    tag = Tag(title=m.text)
    tag.query=q
    tag.save()
    for d in mdocs:
        d.tag.add(tag)


clusters = [6,11,12,15,17,23,27,34,39,67]

ntitle = "fast_greedy_weighted_2017_07_19"

for c in clusters:
    mdocs = Doc.objects.filter(
        query=q,
        networkproperties__network__title=ntitle,
        networkproperties__value=c
    )
    tag = Tag(title="cluster {}, {}".format(c,ntitle))
    tag.query=q
    tag.save()
    for d in mdocs:
        d.tag.add(tag)
