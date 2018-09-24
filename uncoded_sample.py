import django, os, sys, time, resource, re, gc, shutil
from django.db.models import Count, Avg, F
import random
import datetime
sys.path.append('/home/galm/software/tmv/BasicBrowser/')
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *

q = Query.objects.get(pk=1457)
uncoded_docs = Doc.objects.filter(query=q,content__iregex='\w').exclude(docownership__relevant__gt=0)
uncoded_docs_nosus = Doc.objects.filter(
    query=q,content__iregex='\w'
).exclude(
    docownership__relevant__gt=0
).exclude(
    wosarticle__so__icontains="sustainab"
)

wos_sus = Doc.objects.filter(
    query=q,
    wosarticle__kwp__icontains="sustainab"
).exclude(
    title__icontains="sustainab"
).exclude(
    content__icontains="sustainab"
).exclude(
    wosarticle__de="sustainab"
).values_list('pk',flat=True)

t753 = Doc.objects.filter(tag__id=753).values_list('pk',flat=True)

uncoded_docs_nosus = uncoded_docs_nosus.exclude(id__in=list(wos_sus)+list(t753))

s_ids = random.sample(list(uncoded_docs_nosus.values_list('pk',flat=True)),1000)
new_docs = Doc.objects.filter(id__in=s_ids)
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
tag = Tag(
    title="random sample {}".format(now),
    query=q
)
tag.save()
uc = q.users.count()
for i,d in enumerate(new_docs):
    d.tag.add(tag)
    u = q.users.all()[i % uc]
    #for u in q.users.all():
    do = DocOwnership(
        doc=d,
        user=u,
        query=q,
        tag=tag
    )
    do.save()
