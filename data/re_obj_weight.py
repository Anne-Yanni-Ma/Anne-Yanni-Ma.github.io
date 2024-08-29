import torch
import numpy as np
from torch.utils.data import Dataset

with open('/home/yma3/pro/KISGP_Hier/data/classes_leaf.txt', 'r') as f:
    classes_hier = f.readlines()
    for i in range(len(classes_hier)):
        classes_hier[i] = classes_hier[i].strip()
f.close()

with open('/home/yma3/pro/KISGP_Hier/data/classes.txt', 'r') as f:
    classes = f.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].strip()
f.close()


obj_w = np.load('/home/yma3/pro/KISGP_Hier/data/obj_w.npy')
obj_w_hier = np.ones(160)
print("obj_w ", obj_w)
print("classes_hier ",classes_hier)
print("classes ",classes)
for i in range(0,len(obj_w)):
    if (obj_w[i]==0.25):
        print("index: ",i, ", class: ",classes[i])
        #index_MLC = np.where(classes_MLC==classes[i])
        index_hier = classes_hier.index(classes[i])
        print("index_hier: ", index_hier,", classes_hier: ", classes_hier[index_hier])
        obj_w_hier[index_hier] = 0.25
print(obj_w_hier)
np.save("/home/yma3/pro/KISGP_Hier/data/obj_w_hier.npy",obj_w_hier)
print("Reorder object weight, Done!")



