import ens,learn


#def exo():

def single_exp(common,binary=None):
    ensemble=ens.Ensemble()
    if(binary is None):
    	result=learn.train_model(common,
    		                     binary=False,
    		                      clf_type="LR")
    else:
        result,_=ensemble((common,binary))
    print(result.get_acc())


if __name__ == "__main__":
    path="3DHOI/%s/feats"
    common=["shapelets"]
    common=[ path % common_i for common_i in common]
#    common=[path % "1D_CNN"]#,"3DHOI/feats"]
    binary=path % "ens/splitI/"
    exp(common,binary)
