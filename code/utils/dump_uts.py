import django, sys, os

sys.stdout.flush()

# import file for easy access to browser database
sys.path.append('/home/galm/software/tmv/BasicBrowser/')

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BasicBrowser.settings")
django.setup()

from scoping.models import *

qid = 1457

q = Query.objects.get(pk=qid)

docs = Doc.objects.filter(query=q).distinct('UT').values_list('UT',flat=True)

with open("data/1457.txt","w") as f:
    for d in docs:
        f.write(d)
        f.write('\n')
