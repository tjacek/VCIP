import os,os.path
import files

def rename_fun(in_path):
    for path_i in files.top_files(in_path):
        new_path_i=get_new_path(path_i)
        print(path_i)
        print(new_path_i)
        os.rename(path_i,new_path_i)

def get_new_path(path_i):
    dir_i,name_i=os.path.split(path_i)
    name_i,postfix=name_i.split(".")
    name_i="_".join(name_i.split("_")[:-1])
    new_path_i= "%s/%s.%s" % (dir_i,name_i,postfix)
    return new_path_i

rename_fun("3DHOI/seqs")