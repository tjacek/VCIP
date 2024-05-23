import ens,learn


def exp(common,binary):
    lines=[]
    acc_i=single_exp(None,binary)
    lines.append(f"-,{acc_i:.4f}")
    for common_i in common:
        id_i=common_i.split("/")[-2]
        acc_i=single_exp(common_i,binary=None)
        lines.append(f"{id_i},No,{acc_i:.4f}")
        acc_i=single_exp(common_i,binary=binary)
        lines.append(f"{id_i},Yes,{acc_i:.4f}")
    acc_i=single_exp(common,binary=None)
    lines.append(f"full,No,{acc_i:.4f}")
    acc_i=single_exp(common,binary=binary)
    lines.append(f"full,Yes,{acc_i:.4f}")
    print("\n".join(lines))

def single_exp(common,binary=None):
    ensemble=ens.Ensemble()
    if(binary is None):
    	result=learn.train_model(common,
    		                     binary=False,
    		                      clf_type="LR")
    else:
        result,_=ensemble((common,binary))
    return result.get_acc()


if __name__ == "__main__":
    path="3DHOI/%s/feats"
    common=[ "1D_CNN","shapelets"]
    common=[ path % common_i for common_i in common]
#    common=[path % "1D_CNN"]#,"3DHOI/feats"]
    binary=path % "ens/splitI/"
    exp(common,binary)
