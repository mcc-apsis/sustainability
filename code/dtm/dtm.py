#!/usr/bin/env python3

import sys, resource, os, shutil, re, string, gc, subprocess
import django
import nltk
from multiprocess import Pool
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from time import time, sleep
from functools import partial
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix, find
import numpy as np


# Import django stuff
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

# sys.path.append('/home/max/Desktop/django/BasicBrowser/')
import db as db
from tmv_app.models import *
from scoping.models import Doc, Query
from django.db import connection, transaction
cursor = connection.cursor()

def f_gamma2(docs,gamma,docsizes,docUTset,topic_ids):
    vl = []
    for d in docs:
        if gamma[2][d] > 0.001:
            dt = (
                docUTset[gamma[0][d]],
                topic_ids[gamma[1][d]],
                gamma[2][d],
                gamma[2][d] / docsizes[gamma[0][d]],
                run_id
            )
            vl.append(dt)
    return vl

def tokenize(text):
    transtable = {ord(c): None for c in string.punctuation + string.digits}
    tokens = nltk.word_tokenize(text.translate(transtable))
    tokens = [i for i in tokens if len(i) > 2]
    return tokens


def add_features(title):
    django.db.connections.close_all()
    term, created = Term.objects.get_or_create(title=title)
    term.run_id.add(run_id)
    django.db.connections.close_all()
    return term.pk

class snowball_stemmer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer("english")
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in tokenize(doc)]

def proc_docs(docs):
    stoplist = set(nltk.corpus.stopwords.words("english"))
    stoplist.add('elsevier')
    stoplist.add('rights')
    stoplist.add('reserved')
    stoplist.add('john')
    stoplist.add('wiley')
    stoplist.add('sons')
    stoplist.add('copyright')

    abstracts = [re.split("\([C-c]\) [1-2][0-9]{3} Elsevier",x.content)[0] for x in docs.iterator()]
    abstracts = [x.split("Published by Elsevier")[0] for x in abstracts]
    abstracts = [x.split("Copyright (C)")[0] for x in abstracts]
    abstracts = [re.split("\. \(C\) [1-2][0-9]{3} ",x)[0] for x in abstracts]
    docsizes = [len(x) for x in abstracts]
    ids = [x.UT for x in docs.iterator()]
    PYs = [x.PY for x in docs.iterator()]

    return [abstracts, docsizes, ids, stoplist, PYs]

def readInfo(p):
    d = {}
    with open(p) as f:
        for line in f:
            (key, val) = line.strip().split(' ',1)
            try:
                d[key] = int(val)
            except:
                d[key] = val
    return(d)

def dtm_topic(topic_n,info,topic_ids,vocab_ids,ys):
    print(topic_n)
    django.db.connections.close_all()
    p = "%03d" % (topic_n,)
    p = "dtm-output/lda-seq/topic-"+p+"-var-e-log-prob.dat"
    tlambda = np.fromfile(p, sep=" ").reshape((info['NUM_TERMS'],info['SEQ_LENGTH']))
    for t in range(len(tlambda)):
        for py in range(len(tlambda[t])):
            score = np.exp(tlambda[t][py])
            if score > 0.001:
                tt = TopicTerm(
                    topic_id = topic_ids[topic_n],
                    term_id = vocab_ids[t],
                    PY = ys[py],
                    score = score,
                    run_id=run_id
                )
                tt.save()
                #db.add_topic_term(topic_n+info['first_topic'], t+info['first_word'], py, score)
    django.db.connections.close_all()

#########################################################
## Main function

def main():
    try:
        qid = int(sys.argv[1])
    except:
        print("please provide a query ID!")
        sys.exit()

    sleep(7200)
    K = 80
    n_features=20000

    global run_id
    run_id = db.init(n_features,1)

    stat = RunStats.objects.get(pk=run_id)
    stat.method='BD'
    stat.save()
    stat.query=qid

    ##########################
    ## create input folder

    if (os.path.isdir('dtm-input')):
    	shutil.rmtree('dtm-input')

    os.mkdir('dtm-input')

    yrange = list(range(1990,2017))
    #yrange = list(range(1990,1997))

    docs = Doc.objects.filter(
        query=qid,
        content__iregex='\w',
        PY__in=yrange
    ).order_by('PY')

    abstracts, docsizes, ids, stoplist, PYs = proc_docs(docs)

    #########################
    ## Get the features now
    print("Extracting tf-idf features for NMF...")
    vectorizer = CountVectorizer(max_df=0.95, min_df=10,
                                       max_features=n_features,
                                       ngram_range=(1,1),
                                       tokenizer=snowball_stemmer(),
                                       stop_words=stoplist)
    t0 = time()
    dtm = vectorizer.fit_transform(abstracts)

    print("done in %0.3fs." % (time() - t0))

    del abstracts

    gc.collect()

    # Get the vocab, add it to db
    vocab = vectorizer.get_feature_names()
    vocab_ids = []
    pool = Pool(processes=8)
    vocab_ids.append(pool.map(add_features,vocab))
    pool.terminate()
    del vocab
    vocab_ids = vocab_ids[0]

    django.db.connections.close_all()

    with open('dtm-input/foo-mult.dat','w') as mult:
        for d in range(dtm.shape[0]):
            words = find(dtm[d])
            uwords = len(words[0])
            mult.write(str(uwords) + " ")
            for w in range(uwords):
                index = words[1][w]
                count = words[2][w]
                mult.write(str(index)+":"+str(count)+" ")
            mult.write('\n')


    ##########################
    ##put PY stuff in the seq file

    ycounts = docs.values('PY').annotate(
        count = models.Count('pk')
    )

    with open('dtm-input/foo-seq.dat','w') as seq:
        seq.write(str(len(yrange)))

        for y in ycounts:
            seq.write('\n')
            seq.write(str(y['count']))

    ##########################
    # Run the dtm
    subprocess.Popen([
        "/home/galm/software/dtm/dtm/main",
        "--ntopics={}".format(K),
        "--mode=fit",
        "--rng_seed=0",
        "--initialize_lda=true",
        "--corpus_prefix=/home/galm/projects/sustainability/dtm-input/foo",
        "--outname=/home/galm/projects/sustainability/dtm-output",
        "--top_chain_var=0.005",
        "--alpha=0.01",
        "--lda_sequence_min_iter=6",
        "--lda_sequence_max_iter=10",
        "--lda_max_em_iter=10"
    ]).wait()




    ##########################
    ## Upload the dtm results to the db

    info = readInfo("dtm-output/lda-seq/info.dat")

    topic_ids = db.add_topics(K)

    #################################
    # TopicTerms

    topics = range(info['NUM_TOPICS'])
    pool = Pool(processes=8)
    pool.map(partial(
        dtm_topic,
        info=info,
        topic_ids=topic_ids,
        vocab_ids=vocab_ids,
        ys = yrange
    ),topics)
    pool.terminate()
    gc.collect()


    ######################################
    # Doctopics
    gamma = np.fromfile('dtm-output/lda-seq/gam.dat', dtype=float,sep=" ")
    gamma = gamma.reshape((len(gamma)/info['NUM_TOPICS'],info['NUM_TOPICS']))

    gamma = find(csr_matrix(gamma))
    glength = len(gamma[0])
    chunk_size = 100000
    ps = 16
    parallel_add = True

    all_dts = []

    make_t = 0
    add_t = 0

    def insert_many(values_list):
        query='''
            INSERT INTO "tmv_app_doctopic"
            ("doc_id", "topic_id", "score", "scaled_score", "run_id")
            VALUES (%s,%s,%s,%s,%s)
        '''
        cursor = connection.cursor()
        cursor.executemany(query,values_list)

    for i in range(glength//chunk_size+1):
        dts = []
        values_list = []
        f = i*chunk_size
        l = (i+1)*chunk_size
        if l > glength:
            l = glength
        docs = range(f,l)
        doc_batches = []
        for p in range(ps):
            doc_batches.append([x for x in docs if x % ps == p])
        pool = Pool(processes=ps)
        make_t0 = time()
        values_list.append(pool.map(partial(f_gamma2, gamma=gamma,
                        docsizes=docsizes,docUTset=ids,topic_ids=topic_ids),doc_batches))
        #dts.append(pool.map(partial(f_gamma, gamma=gamma,
        #                docsizes=docsizes,docUTset=ids,topic_ids=topic_ids),doc_batches))
        pool.terminate()
        make_t += time() - make_t0
        django.db.connections.close_all()

        add_t0 = time()
        values_list = [item for sublist in values_list for item in sublist]
        pool = Pool(processes=ps)
        pool.map(insert_many,values_list)
        pool.terminate()
        add_t += time() - add_t0
        gc.collect()
        sys.stdout.flush()




if __name__ == '__main__':
    t0 = time()
    main()
    totalTime = time() - t0

    tm = int(totalTime//60)
    ts = int(totalTime-(tm*60))

    print("done! total time: " + str(tm) + " minutes and " + str(ts) + " seconds")
    print("a maximum of " + str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1000) + " MB was used")
