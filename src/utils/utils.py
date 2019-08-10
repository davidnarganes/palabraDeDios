import os

def mknewdir(newdir):
    if os.path.exists(newdir):
        print("'%s' already exists" % newdir)
    else:
        os.mkdir(newdir)
        print("'%s' created" % newdir)