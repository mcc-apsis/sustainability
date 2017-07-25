with open("/queries/sus_all/results.txt","w") as wf:
    with open("/queries/sus_all/results_1.txt","r") as r:
        for l in r:
            if "FN Clarivate Analytics Web of Science" not in l:
                wf.write(l)
                wf.write('\n')
    with open("/queries/sus_all/results_2.txt","r") as r:
        for l in r:
            if "FN Clarivate Analytics Web of Science" not in l:
                wf.write(l)
                wf.write('\n')
