import numpy as np
import files

class Seqs(dict):
    def __init__(self, arg=[]):
        super(Seqs, self).__init__(arg)

    def split(self,selector=None):
        train,test=files.split(self,selector)
        return Seqs(train),Seqs(test)
        
    def as_dataset(self):
        X,y=[],[]
        names=list(self.keys())
        for name_i in names:
            X.append(self[name_i])
            y.append(name_i.get_cat())
        return np.array(X),y,names

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