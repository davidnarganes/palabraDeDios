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
<<<<<<< HEAD
    return re.sub("\W+", str(d), "_")
=======
    return re.sub("\W+", str(d), "_")

def unnest(nested_list):
    '''
    flattens a list of lists
    '''
    return [e for l in nested_list for e in l]
>>>>>>> b61cd0979d7885a7b9003b1ed7edac8bc54c781a
