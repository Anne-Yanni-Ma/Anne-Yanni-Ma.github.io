import numpy as np
import torch
import torch.nn as nn
import pathmagic  # noqa
from models.pred_classification.pointnet import PointNetEncoderwoBN
from models.pred_classification.utils import FocalLoss, MLP
from pointnet2_ops.pointnet2_utils import FurthestPointSampling


def fps_sampling(xyz, npoints):
    # xyz: B, N, 3
    fps = FurthestPointSampling()
    idx = fps.apply(xyz, npoints).long()      # B, N
    return idx.long()


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.latent_dimension = 512
        self.obj_pointnet_lv1 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_pointnet_lv2 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_pointnet_lv3 = PointNetEncoderwoBN(transform=False, in_channel=3, out_channel=self.latent_dimension)
        self.obj_mlp = MLP(mlp=[512*3, 1024, 1024, 512])
        self.hier_obj_classifer = HierarchicalClassifier()#level1_classes, level2_classes_dict, level3_classes_dict)

    def forward(self, pc_mat):
        insnum = pc_mat.shape[0]
        pc_mat = self.normalize_regularization(pc_mat) #(B,N,3)
        if self.training:
            pc_mat = torch.bmm(pc_mat, self.random_rotation(insnum))
        idx_256 = fps_sampling(pc_mat, 256)
        idx_128 = fps_sampling(pc_mat, 128)
        pc_256 = torch.cat([pc_mat[i, idx_256[i], :].view(1, 256, 3) for i in range(insnum)], dim=0).contiguous()
        pc_128 = torch.cat([pc_mat[i, idx_128[i], :].view(1, 128, 3) for i in range(insnum)], dim=0).contiguous()
        obj_codes_lv1 = self.obj_pointnet_lv1(pc_mat.transpose(2, 1).contiguous())
        obj_codes_lv2 = self.obj_pointnet_lv2(pc_256.transpose(2, 1).contiguous())
        obj_codes_lv3 = self.obj_pointnet_lv3(pc_128.transpose(2, 1).contiguous())
        obj_codes = torch.cat([obj_codes_lv1, obj_codes_lv2, obj_codes_lv3], dim=1)
        obj_codes = self.obj_mlp(obj_codes)

        hier_obj_output = self.hier_obj_classifer(obj_codes)
        level1_out, level2_out, level3_output = hier_obj_output
        return level3_output, hier_obj_output, obj_codes

    def normalize_regularization(self, pc_mat):
        maxs = torch.max(pc_mat, dim=1, keepdim=True)[0]
        mins = torch.min(pc_mat, dim=1, keepdim=True)[0]
        offsets = (maxs + mins) / 2
        scale = torch.max((maxs - mins), dim=2)[0].view(-1, 1, 1).contiguous()
        pc_mat -= offsets
        pc_mat /= scale
        return pc_mat

    def random_rotation(self, insnum):
        rm = []
        for i in range(insnum):
            rotation_angle = np.random.uniform(-1, 1) * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
            rotation_matrix = torch.Tensor(rotation_matrix).cuda()
            rm.append(rotation_matrix)
        rm = torch.stack(rm, dim=0)
        return rm



class HierarchicalClassifier(nn.Module):
    def __init__(self):
        super(HierarchicalClassifier, self).__init__()
        self.obj_mlp_l1 = MLP(mlp=[512,512, 512])
        self.obj_mlp_l2 = MLP(mlp=[512,512, 512])
        self.obj_mlp_l3 = MLP(mlp=[512,512, 512])

        self.level1_classifier = MLP(mlp=[512, 256, 128, 5])
        self.level2_classifier = MLP(mlp=[512, 256, 128, 19])
        self.level3_classifier = MLP(mlp=[512, 256, 256, 160])

    def forward(self, x):
        obj_codes1= obj_codes2=obj_codes3 = x
        '''output_l1 = self.level1_classifier(self.obj_mlp_l1(obj_codes1))
        output_l2 = self.level2_classifier(self.obj_mlp_l2(obj_codes2))
        output_l3 = self.level3_classifier(self.obj_mlp_l3(obj_codes3))'''
        output_l1 = self.level1_classifier(x)
        output_l2 = self.level2_classifier(x)
        output_l3 = self.level3_classifier(x)
        hier_output = output_l1, output_l2, output_l3
        return hier_output

level1_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
level2_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18]
level3_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]

class HierarchicalLoss(nn.Module):
    def __init__(self,gamma, obj_w):
        super(HierarchicalLoss, self).__init__()
        self.focal_loss_l1 = FocalLoss(class_num=5, alpha=None, gamma=gamma, size_average=True)
        self.focal_loss_l2 = FocalLoss(class_num=19, alpha=None, gamma=gamma, size_average=True)
        self.focal_loss_l3 = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True)

    def forward(self, hier_pred, gt_obj):
        pred_l1, pred_l2, pred_l3 = hier_pred

        gt_obj_l3 = self.prepare_objgt(gt_obj)
        obj_loss_l3 = self.focal_loss_l3(pred_l3, gt_obj_l3)  # focal_loss

        gt_obj_l2 = self.prepare_hier_objgt(gt_obj, level2_target,19)
        obj_loss_l2 = self.focal_loss_l2(pred_l2, gt_obj_l2)

        gt_obj_l1 = self.prepare_hier_objgt(gt_obj,level1_target,5)
        obj_loss_l1 = self.focal_loss_l1(pred_l1, gt_obj_l1)
        obj_loss = obj_loss_l1 + obj_loss_l2 + obj_loss_l3
        obj_loss =  obj_loss_l2 + obj_loss_l3

        return obj_loss

    def prepare_hier_objgt(self, obj_gt, target,label_num):
        insnum = obj_gt.shape[0]
        onehot = torch.zeros(insnum, label_num).float().cuda()
        for i in range(insnum):
            onehot[i, target[obj_gt[i]]] = 1
        return onehot
    def prepare_objgt(self, obj_gt):
        insnum = obj_gt.shape[0]
        onehot = torch.zeros(insnum, 160).float().cuda()
        for i in range(insnum):
            onehot[i, obj_gt[i]] = 1
        return onehot

