import os
import re

def mknewdir(newdir):
    print("Guillermo mola cachote")
    if os.path.exists(newdir):
        print("`%s` already exists" % newdir)
    else:
        os.mkdir(newdir)
        print("`%s` created" % newdir)

def dict2str(d):
    """
    Convert a dict of args into a str
    """
    return re.sub("\W+", "_", str(d))

def unnest(nested_list):
    '''
    flattens a list of lists
    '''
    return [e for l in nested_list for e in l]
