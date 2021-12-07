import numpy as np
import feats,learn

class Ensemble(object):
    def __init__(self,read=None):
        if(read is None):
            read=read_dataset
        self.read=read

    def __call__(self,paths,binary=False,clf="LR"):
        datasets=self.get_datasets(paths)
        votes=make_votes(datasets,clf=clf)
        result=votes.voting(binary)
        print(result.get_acc()) 
        return result,votes

    def get_datasets(self,paths):
        if(type(paths)==dict):
            common,binary=paths["common"],paths["binary"]
        else:
            common,binary=paths    
        datasets=self.read(common,binary)
        return datasets

class Votes(object):
    def __init__(self,results):
        self.results=results

    def __len__(self):
        return len(self.results)

    def voting(self,binary=False):
        if(binary):
            votes=np.array([ result_i.as_hard_votes() 
                    for result_i in self.results])
        else:
            votes=np.array([ result_i.as_numpy() 
                    for result_i in self.results])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def weighted(self,weights):
        votes=np.array([ weight_i*result_i.as_numpy() 
                    for weight_i,result_i in zip(weights,self.results)])
        votes=np.sum(votes,axis=0)
        return learn.Result(self.results[0].y_true,votes,self.results[0].names)

    def get_acc(self):
        return [ result_i.get_acc() for result_i in self.results]

def read_dataset(common_path,deep_path):
    if(not common_path):
        return read_deep(deep_path)
    if(not deep_path):
        return feats.read(common_path)
    common_data=feats.read(common_path)[0]
    deep_data=read_deep(deep_path)
    datasets=[common_data+ data_i 
                for data_i in deep_data]
    return datasets

def read_deep(deep_path):
    if(type(deep_path)==list):
        datasets=[]
        for deep_i in deep_path:
            datasets+=feats.read(deep_i)
        return datasets
    return feats.read(deep_path)

def make_votes(datasets,clf="LR"):    
    results=[learn.train_model(data_i,clf_type=clf,binary=False)
                    for data_i in datasets]
    return Votes(results)  

in_path="3DHOI/1D_CNN/feats"
ensemble=Ensemble()
ensemble((in_path,None))