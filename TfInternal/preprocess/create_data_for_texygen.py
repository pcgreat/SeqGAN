import pdb
from collections import Counter
from random import shuffle

from six.moves import cPickle

ltoks = []
with open("zuowen.txt","w") as g:
    with open("../data/zuowen_sents.txt", "r") as f:
        for line in f:
            toks = list(line.strip().replace(" ", ""))
            g.write(" ".join(toks) +"\n")