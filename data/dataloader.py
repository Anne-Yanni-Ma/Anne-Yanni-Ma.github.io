import torch
import numpy as np
from torch.utils.data import Dataset
#from vtkplotter import show, Points
#class_to_leaf_idx = {0: 57, 1: 33, 2: 34, 3: 145, 4: 69, 5: 27, 6: 35, 7: 106, 8: 26, 9: 83, 10: 68, 11: 58, 12: 28, 13: 36, 14: 84, 15: 96, 16: 95, 17: 153, 18: 154, 19: 88, 20: 48, 21: 37, 22: 146, 23: 38, 24: 107, 25: 87, 26: 118, 27: 59, 28: 39, 29: 17, 30: 108, 31: 124, 32: 0, 33: 147, 34: 1, 35: 70, 36: 109, 37: 71, 38: 60, 39: 72, 40: 73, 41: 49, 42: 110, 43: 97, 44: 85, 45: 133, 46: 128, 47: 74, 48: 61, 49: 75, 50: 114, 51: 115, 52: 89, 53: 40, 54: 2, 55: 98, 56: 99, 57: 119, 58: 129, 59: 130, 60: 155, 61: 151, 62: 62, 63: 140, 64: 134, 65: 41, 66: 42, 67: 131, 68: 18, 69: 100, 70: 148, 71: 125, 72: 50, 73: 19, 74: 3, 75: 111, 76: 76, 77: 4, 78: 101, 79: 21, 80: 12, 81: 29, 82: 22, 83: 5, 84: 90, 85: 137, 86: 6, 87: 103, 88: 13, 89: 138, 90: 77, 91: 149, 92: 150, 93: 91, 94: 63, 95: 7, 96: 51, 97: 135, 98: 53, 99: 156, 100: 14, 101: 157, 102: 158, 103: 159, 104: 86, 105: 141, 106: 132, 107: 136, 108: 20, 109: 43, 110: 15, 111: 92, 112: 102, 113: 44, 114: 8, 115: 64, 116: 78, 117: 16, 118: 93, 119: 126, 120: 94, 121: 127, 122: 112, 123: 104, 124: 105, 125: 120, 126: 121, 127: 79, 128: 30, 129: 52, 130: 142, 131: 65, 132: 66, 133: 122, 134: 80, 135: 67, 136: 9, 137: 152, 138: 31, 139: 81, 140: 23, 141: 24, 142: 25, 143: 32, 144: 144, 145: 54, 146: 55, 147: 139, 148: 45, 149: 46, 150: 143, 151: 10, 152: 82, 153: 56, 154: 123, 155: 113, 156: 11, 157: 47, 158: 116, 159: 117}

class_to_leaf_idx = [70, 33, 34, 145, 89, 27, 35, 62, 26, 57, 88, 71, 28, 36, 58, 104, 103, 153, 154, 81, 48, 37, 146, 38, 63, 61, 119, 72, 39, 17, 64, 125, 0, 147, 1, 90, 65, 91, 73, 92, 93, 49, 66, 105, 59, 134, 129, 94, 74, 95, 107, 108, 82, 40, 2, 106, 109, 120, 130, 131, 155, 148, 75, 110, 135, 41, 42, 132, 18, 111, 149, 126, 50, 19, 3, 67, 96, 4, 112, 21, 12, 29, 22, 5, 83, 138, 6, 114, 13, 139, 97, 150, 151, 84, 76, 7, 51, 136, 53, 156, 14, 157, 158, 159, 60, 141, 133, 137, 20, 43, 15, 85, 113, 44, 8, 77, 98, 16, 86, 127, 87, 128, 68, 115, 116, 121, 122, 99, 30, 52, 142, 78, 79, 123, 100, 80, 9, 152, 31, 101, 23, 24, 25, 32, 144, 54, 55, 140, 45, 46, 143, 10, 102, 56, 124, 69, 11, 47, 117, 118]

def map_gt_obj_to_leaf(gt_obj):
    # 将 gt_obj 中的索引从 classes 映射到 classes_leaf
    gt_obj_leaf = [class_to_leaf_idx[idx] for idx in gt_obj]
    return gt_obj_leaf

with open('./data/classes.txt', 'r') as f:
    classes = f.readlines()
    for i in range(len(classes)):
        classes[i] = classes[i].strip()
f.close()


with open('./data/relationships.txt', 'r') as f:
    relationships = f.readlines()
    for i in range(len(relationships)):
        relationships[i] = relationships[i].strip()
f.close()


def visualize(mat):
    insnum = mat.shape[0]
    rand_color = np.random.rand(insnum, 3)
    pc_mat = []
    color_mat = []
    for i in range(insnum):
        pc_mat.append(mat[i])
        c = rand_color[i].reshape((1, 3)).repeat(512, axis=0)
        color_mat.append(c)
    pc_mat = np.vstack(pc_mat)
    color_mat = np.vstack(color_mat)
    pc = Points(pc_mat, c=color_mat)
    show(pc, interactive=1)


class DataLoader_3DSSG(Dataset):
    def __init__(self, training=True, shuffle=False, norm=False, half=False, per25=False):
        self.training = training
        self.norm = norm

        if shuffle:#False
            self.training_txt = 'XX'
            self.test_txt = 'XX'
        else:
            if half:#False
                self.training_txt = '/home/yma3/dataset/3DSSG_subset/train/training_txt.txt'
            elif per25:#True
                self.training_txt = '/home/yma3/dataset/3DSSG_subset/train/training_txt.txt'
            else:
                self.training_txt = '/home/yma3/dataset/3DSSG_subset/train/training_txt.txt'
            #self.test_txt = '/home/ma/dataset/3DSSG/standard_split_sampled/test/testing_txt.txt'
            self.test_txt = "/home/yma3/dataset/3DSSG_subset/test/testing_txt.txt" # sampled5
            #self.test_txt = "/home/ma/myn/2022-new/know-3DSSG/data/3DSSG_data_examples/testing_txt.txt" # sampled5


        self.training_list = []
        self.test_list = []

        with open(self.training_txt, 'r') as f:
            self.training_list = f.readlines()
            for i in range(len(self.training_list)):
                self.training_list[i] = self.training_list[i].strip()
        f.close()

        with open(self.test_txt, 'r') as f:
            self.test_list = f.readlines()
            for i in range(len(self.test_list)):
                self.test_list[i] = self.test_list[i].strip()
        f.close()

        self.training_len = len(self.training_list)
        self.testing_len = len(self.test_list)
        self.obj_w = torch.Tensor(np.load('./data/obj_w_hier.npy')).cuda()
        self.pred_w = torch.Tensor([0.25, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1]).cuda()

    def __len__(self):
        if self.training:
            return self.training_len
        else:
            return self.testing_len

    def __getitem__(self, index):
        if self.training:
            folder = self.training_list[index]
        else:
            folder = self.test_list[index]
        obj_gt = np.load(folder +'/gt_obj.npy')
        obj_gt_LEAF = map_gt_obj_to_leaf(obj_gt)
        obj_gt = obj_gt_LEAF
        rel_gt = np.load(folder + '/gt_relationships.npy')  # (25, 3)
        #print(rel_gt)
        pc_mat = np.load(folder + '/pointcloud_1024_ins.npy')[:, :, 0:3]
        #print(pc_mat)
        if self.norm:
            pc_mat = self.normalize(pc_mat)
        return torch.Tensor(pc_mat), torch.IntTensor(obj_gt), torch.IntTensor(rel_gt)

    def visualize(self, index):
        if self.training:
            folder = self.training_list[index]
        else:
            folder = self.test_list[index]
        obj_gt = np.load(folder + '/gt_obj.npy')
        rel_gt = np.load(folder + '/gt_relationships.npy')
        pc_mat = np.load(folder + '/pointcloud_1024_ins.npy')[:, :, 0:6]
        for i in range(rel_gt.shape[0]):
            print(classes[obj_gt[rel_gt[i, 0]]] + '->' + classes[obj_gt[rel_gt[i, 1]]] + '=' + relationships[rel_gt[i, 2]])
        pc_mat = pc_mat.reshape(-1, 6)
        pc = Points(pc_mat[:, 0:3], c=pc_mat[:, 3:6])
        show(pc, interactive=1)

    def normalize(self, pc_mat):
        xyz = pc_mat[:, :, 0:3]
        maxs = np.max(np.max(xyz, axis=0), axis=0)
        mins = np.min(np.min(xyz, axis=0), axis=0)
        offsets = (maxs + mins) / 2
        scale = (maxs - mins).max()
        pc_mat[:, :, 0:3] -= offsets
        pc_mat[:, :, 0:3] /= scale
        mins = np.min(np.min(xyz, axis=0), axis=0)
        mins[0] = 0
        mins[1] = 0
        pc_mat[:, :, 0:3] -= mins
        return pc_mat


if __name__ == "__main__":
    dataset3dssg_train = DataLoader_3DSSG(training=False)
    pc_mat, obj_gt, rel_gt = dataset3dssg_train.__getitem__(173)
    print(obj_gt)
    print(rel_gt)
    dataset3dssg_train.visualize(173)
    print(dataset3dssg_train.obj_w)
