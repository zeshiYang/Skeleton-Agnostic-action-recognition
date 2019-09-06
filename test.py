import os
import numpy as np

def exist(a,b):
    for i in a:
        if(i==b):
            return True
    return False


path ="../data/"
files = os.listdir(path)
id_list=[]
for i in range(len(files)):
    actor_index = int(files[i][9:12])
    if(not exist(id_list,actor_index)):
        id_list.append(actor_index)

print(id_list)
