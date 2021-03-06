import numpy as np
import random
from scipy.interpolate import CubicSpline
import files,feats

class Seqs(dict):
    def __init__(self, arg=[]):
        super(Seqs, self).__init__(arg)

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return Seqs(train),Seqs(test)
    
    def random_subset(self,k):
        names=list(self.keys())
        random.shuffle(names)
        return self.subset(names[:k])

    def subset(self,names):
        seq_dict=Seqs()
        for name_i in names:
            seq_dict[name_i]=self[name_i]
        return seq_dict

    def resize(self,new_size=32):
        for name_i in self.keys():
            self[name_i]=inter(self[name_i],new_size)

    def as_dataset(self):
        X,y=[],[]
        names=list(self.keys())
        for name_i in names:
            X.append(self[name_i])
            y.append(name_i.get_cat())
        return np.array(X),y,names

    def transform(self,fun):
        for name_i in self.keys():
            self[name_i]=fun(self[name_i])

    def save(self,out_path):
        files.make_dir(out_path)
        for name_i,seq_i in self.items():
            print(seq_i.shape)
            out_i="%s/%s" % (out_path,name_i)
            np.save(out_i,seq_i)

def read_seqs(in_path):
    seqs=Seqs()
    for path_i in files.top_files(in_path):
        data_i=read_data(path_i)
        name_i=path_i.split('/')[-1]
        name_i=files.Name(name_i).clean()
        seqs[name_i]=data_i
    return seqs

def read_data(path_i):
    if(is_npy(path_i)):
        return np.load(path_i)
    else:
        return np.loadtxt(path_i, delimiter=',')

def is_npy(path_i):
    return path_i.split(".")[-1]=="npy"

def inter(ts_i,new_size):
    old_size=ts_i.shape[0]
    step= new_size/old_size
    old_x=np.arange(old_size).astype(float)
    old_x*=step
    cs=CubicSpline(old_x,ts_i)
    new_x=np.arange(new_size)
    return cs(new_x)

def to_feats(in_path,fun,out_path=None):
    new_feats=feats.Feats()
    for path_i in files.top_files(in_path):
        data_i=read_data(path_i)
        name_i=files.get_name(path_i)
        new_feats[name_i]=fun(name_i,data_i)
    if(out_path):
        new_feats.save(out_path)
    return new_feats