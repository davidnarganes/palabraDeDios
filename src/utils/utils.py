import os
import re
from itertools import islice

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

def read_bible(in_filepath):
    with open(in_filepath, "r",  encoding='latin1') as infile:
        lines = infile.read()
    return lines.split("\t")

def make_windows(list_, window_width=30):
    """
    Function to take a list and yield sublists of a given window size

    Args:
        - list_
        - window_width
    
    Returns:
        - yield generator of a sliding window from list_
            s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    """
    
    it = iter(list_)
    result = tuple(islice(it, window_width))
    if len(result) == window_width:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
