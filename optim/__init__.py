import sys
sys.path.append("..")
import diff_evol,gasen,ens#,exp

class MultiEnsembleExp(object):
    def __init__(self,all_ensembles,threshold=0.05):
        self.all_ensembles=all_ensembles
        self.threshold=threshold

    def __call__(self,input_dict,out_path=None):
        lines=[]
        for desc_i,ensemble_i in self.all_ensembles.items():
            result_i,votes_i=ensemble_i(input_dict)
            if(type(votes_i)==ens.Votes):
                n_clf=len(votes_i)
            else:
                n_clf=votes_i[votes_i>self.threshold].shape[0] 
            line_i="%s,%d,%s" % (desc_i,n_clf,get_metrics(result_i))
            lines.append(line_i)
        save_lines(lines,out_path)
        return lines

def get_metrics(result_i):
    acc_i= result_i.get_acc()
    metrics="%.4f,%.4f,%.4f" % result_i.metrics()[:3]
    return "%.4f,%s" % (acc_i,metrics)

def save_lines(lines,out_path):
    print(lines)
    if(out_path):
        txt="\n".join(lines)
        out_file = open(out_path,"w")
        out_file.write(txt)
        out_file.close()

def optim_exp(common_path,binary_path,out_path=None,read_type=None):
    all_ens={}
    all_voting_methods(all_ens ,read_type=read_type ,desc="")
    ens_exp=MultiEnsembleExp(all_ens)
    input_dict=(common_path,binary_path)
    ens_exp(input_dict,out_path)

def all_voting_methods(all_ens ,read_type=None ,desc=""):
    ensemble=ens.Ensemble(read_type)
    algs={"soft":ensemble,"hard":ens.EnsembleHelper(ensemble,binary=False),
           "diff":diff_evol.OptimizeWeights(read=read_type),
           "gasen":diff_evol.OptimizeWeights(gasen.Corl,maxiter=10,read=read_type) }
    for name_i,alg_i in algs.items():
        all_ens["%s%s" % (name_i,desc)]=alg_i
    return all_ens

path="../3DHOI/%s/feats"
common=[path % "1D_CNN","../3DHOI/feats"]
binary=path % "ens/splitI/"
optim_exp(common,binary,out_path="3DHOI.csv",read_type=None)