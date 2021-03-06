import numpy as np
from tslearn.shapelets import LearningShapelets
import timeit
import files,seqs#,feats

def make_feats(in_path,out_path):
    model=train_model(in_path)
    def helper(name_i,data_i):
        print(name_i)
        print(data_i.shape)
        frames=seqs.inter(data_i,36)
        frames= np.squeeze(frames)
        frames=np.expand_dims(frames,axis=0)
        distances = model.transform(frames)
        return distances
    seqs.to_feats(in_path,helper,out_path=out_path)

def train_model(in_path):
    ts=seqs.read_seqs(in_path) #from_paths(paths)
    ts=ts.split()[0]
    ts.resize(36)
    model = LearningShapelets(n_shapelets_per_size=None,max_iter=2) #{3: 40})
    ts=ts.random_subset(k=10)
    X,y,names=ts.as_dataset()
    model.fit(X,y)
    return model

def compute_shaplets(in_path,out_path,n_feats=40):
    ts=seqs.read_seqs(in_path)
    ts.resize(64)
    train,test=ts.split()
    model = LearningShapelets(n_shapelets_per_size={3: n_feats})
    train_X,train_y,train_names=train.as_dataset()
    model.fit(train_X,train_y)
    X,y,names=ts.as_dataset()
    print(X.shape)
    distances = model.transform(X)
    dist_feat=feats.Feats()
    for i,x_i in enumerate(distances):
        dist_feat[names[i]]=x_i
        print(x_i.shape)    
    dist_feat.save(out_path)

in_path="3DHOI/shape_32" 
out_path="3DHOI/feats"
make_feats(in_path,out_path)