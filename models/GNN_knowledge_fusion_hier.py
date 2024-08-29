import torch
import torch.nn as nn
import torch.nn.functional as F
import pathmagic  # noqa
import numpy as np
from utils import FocalLoss, MLP
from gnn_models import GraphEncoderKnowledgeFusion
from graph import SceneGraph
#from hesp.embedding_space.embedding_space import EmbeddingSpace
#from hesp.embedding_space.hyperbolic_embedding_space import HyperbolicEmbeddingSpace



class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()
        self.assimilation = 2
        self.knode = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_node.npy')).cuda()
        self.kedge = torch.Tensor(np.load('./data/meta_embedding/meta_embedding_edge.npy')).cuda()
        self.gnn = GraphEncoderKnowledgeFusion(ndim=512, nlayer=5)
        self.node_mlp = MLP(mlp=[512, 512, 512])
        self.knode_mlp = MLP(mlp=[512, 512, 512])
        self.edge_mlp = MLP(mlp=[512, 512, 512])
        self.kedge_mlp = MLP(mlp=[512, 512, 512])
        self.node_classifer = MLP(mlp=[512, 256, 256, 160])
        self.edge_classifer = MLP(mlp=[512, 256, 128, 27])
        self.conv1d = torch.nn.Conv1d(256, 256, 1)
        self.level1_classifier = MLP(mlp=[512, 256, 128, 5])
        self.level2_classifier = MLP(mlp=[512, 256, 128, 19])
        #self.level3_classifier = MLP(mlp=[512, 256, 256, 160])


    def forward(self, obj_codes, pred_codes): #obj_codes(Nn,C_konw) pred_codes (Ne, C)
        insnum = obj_codes.shape[0] # Nn
        edge_index = self.prepare_edges(insnum) # (2, Nn*(Nn-1))
        g = SceneGraph(x=obj_codes, edge_index=edge_index, edge_attr=pred_codes, edge_weight=torch.ones(pred_codes.shape[0]))  #edge_weight(Ne,)
        knode = torch.zeros(insnum, 512).cuda()    # ([7, 512])
        kedge = torch.zeros(insnum*(insnum-1), 512).cuda()  # ([42, 512])

        for i in range(self.assimilation):
            node_embed, edge_embed = self.gnn(g, knode, kedge)  # ([7, 512])  # ([42, 512])
            node_weight, edge_weight = self.node_classifer(node_embed), self.edge_classifer(edge_embed) # ([7, 160])  # ([42, 27])
            node_weight, edge_weight = F.softmax(node_weight, dim=1), F.softmax(edge_weight, dim=1) #
            knode, kedge = self.select_knowledge(node_weight, edge_weight)
            g.reset_graph()
        output_l1 = F.softmax(self.level1_classifier(node_embed), dim=1)
        output_l2 =  F.softmax(self.level2_classifier(node_embed), dim=1)
        hier_output = output_l1, output_l2, node_weight

        return node_weight, hier_output, edge_weight  # ([7, 160])  # ([42, 27])

    def select_knowledge(self, node_weight, edge_weight):
        node_topk_inds = node_weight.topk(k=5).indices
        node_inds = torch.zeros(node_weight.shape).cuda()
        for i in range(node_topk_inds.shape[0]):
            node_inds[i, node_topk_inds[i]] = 1
        knode = torch.mm(node_inds, self.knode)

        edge_topk_inds = edge_weight.topk(k=5).indices
        edge_inds = torch.zeros(edge_weight.shape).cuda()
        for i in range(edge_topk_inds.shape[0]):
            edge_inds[i, edge_topk_inds[i]] = 1
        kedge = torch.mm(edge_inds, self.kedge)
        return knode, kedge

    def prepare_edges(self, insnum):
        edge_index = torch.zeros(2, insnum*insnum-insnum).long().cuda()
        idx = 0
        for i in range(insnum):
            for j in range(insnum):
                if i != j:
                    edge_index[0, idx] = i
                    edge_index[1, idx] = j
                    idx += 1
        return edge_index


level1_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
level2_target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18]
level3_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]

class HierarchicalLoss(nn.Module):
    def __init__(self,alpha, beta, gamma, obj_w, pred_w):
        super(HierarchicalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_loss_l1 = FocalLoss(class_num=5, alpha=None, gamma=gamma, size_average=True,use_softmax=False)
        self.focal_loss_l2 = FocalLoss(class_num=19, alpha=None, gamma=gamma, size_average=True,use_softmax=False)
        self.focal_loss_l3 = FocalLoss(class_num=160, alpha=obj_w, gamma=gamma, size_average=True,use_softmax=False)
        self.focal_loss_pred = FocalLoss(class_num=27, alpha=pred_w, gamma=gamma, size_average=True, use_softmax=False)

    def forward(self, hier_obj_output, edge_output, gt_obj, gt_rel):
        obj_l1, obj_l2, obj_l3 = hier_obj_output

        gt_obj_l3 = self.prepare_objgt(gt_obj)
        obj_loss_l3 = self.focal_loss_l3(obj_l3, gt_obj_l3)  # focal_loss

        gt_obj_l2 = self.prepare_hier_objgt(gt_obj,level2_target,19)
        obj_loss_l2 = self.focal_loss_l2(obj_l2, gt_obj_l2)

        gt_obj_l1 = self.prepare_hier_objgt(gt_obj,level1_target,5)
        obj_loss_l1 = self.focal_loss_l1(obj_l1, gt_obj_l1)
        obj_loss = obj_loss_l1 + obj_loss_l2 + obj_loss_l3

        predgt_onehot = self.prepare_onehot_predgt(gt_obj, gt_rel)  # ([42, 27])
        pred_loss = self.focal_loss_pred(edge_output, predgt_onehot)
        # print("obj_loss : pred_loss, ", obj_loss, " , ", pred_loss)  #   5~10 : 1
        loss = self.alpha * obj_loss + self.beta * pred_loss  ## alpha:1 beta: 0.1
        return loss

    def prepare_onehot_predgt(self, gt_obj, gt_rel):  # gt_obj (No,), gt_rel(Ne,3)
        insnum = gt_obj.shape[0]
        onehot_gt = torch.zeros((insnum * insnum - insnum, 27)).cuda()

        for i in range(gt_rel.shape[0]):
            idx_i = gt_rel[i, 0]
            idx_j = gt_rel[i, 1]
            if idx_i < idx_j:
                onehot_gt[int(idx_i * (insnum - 1) + idx_j - 1), int(gt_rel[i, 2])] = 1
            elif idx_i > idx_j:
                onehot_gt[int(idx_i * (insnum - 1) + idx_j), int(gt_rel[i, 2])] = 1
        for i in range(insnum * insnum - insnum):
            if torch.sum(onehot_gt[i, :]) == 0:
                onehot_gt[i, 0] = 1
        return onehot_gt

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

