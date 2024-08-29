import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from ..builder import LOSSES
from losses.losses_utils import weight_reduce_loss
from losses.cross_entropy_loss import CrossEntropyLoss
from losses.tree_triplet_loss import TreeTripletLoss
from losses.hiera2_loss import losses_hiera2_focal,losses_hiera2

# node  =   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
'''hiera_map = [0,0,1,1,1,2,2,2,3,3,4,5,5,6,6,6,6,6,6]
hiera_index = [[0,2],[2,5],[5,8],[8,10],[10,11],[11,13],[13,19]]
hiera = {
    "hiera_high":{
        "flat":[0,2],
        "construction":[2, 5],
        "object":[5, 8],
        "nature":[8, 10],
        "sky":[10, 11],
        "human":[11,13],
        "vehicle":[13,19]
    }
}
'''

hiera_map1=[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 , 4 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 5 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 6 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 7 , 8 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 9 , 10 , 10 , 10 , 10 , 10 , 10 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 11 , 12 , 12 , 12 , 13 , 13 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 14 , 15 , 15 , 15 , 16 , 16 , 16 , 16 , 16 , 16 , 16 , 16 , 16 , 16 , 17 , 17 , 17 , 17 , 17 , 18 , 18 , 18 , 18 , 18 , 18 , 18 , 18 , 19 , 19 , 19 , 19 , 19 , 20 , 20 , 20 , 21 , 21 , 22 , 22 , 22 , 22 , 22 , 22 , 23 , 23 , 23 , 23]

hiera_index1 = [[0,8],[8,13],[13,16],[16,25],[25,31],[31,54],[54,67],[67,75],[75,76],[76,84],[84,90],[90,98],[98,101],[101,103],[103,114],[114,117],[117,127],
                [127,132],[132,140],[140,145],[145,148],[148,150],[150,156],[156,160]]

hiera1 = {
    "hiera_high":{
        "device": [0,8],
        "screen_kinds": [8,13],
        "source_of_illumination": [13,16],
        "cleaning": [16,25],
        "cloth_covering": [25,31],
        "container": [31,54],
        "home_appliances": [54,67],
        "paperwork": [67,75],
        "bed": [75,76],
        "cabinets": [76,84],
        "cloth_material": [84,90],
        "plumbing": [90,98],
        "racks": [98,101],
        "shelves": [101,103],
        "sitting": [103,114],
        "stands": [114,117],
        "table_surface": [117,127],
        "construction_surface":  [127,132],
        "structure":[132,140],
        "on_surface": [140,145],
        "entertainment": [145,148],
        "foods": [148,150],
        "others": [150,156],
        "plants": [156,160]
    }
}


hiera_map2=[0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 2 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 3 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4]

hiera_index2 = [[0,16],[16,75],[75,127],[127,145],[145,160]]

hiera2 = {
   "hiera_higher":{
        "general_device": [0,16],
        "general_use": [16,75],
        "house_furniture   ": [75,127],
        "room": [127,145],
        "void": [145,160]
   }
}
# hiera     = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
hiera_map12 = [0,0,0,1,1,1,1,1,2,2,2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4]
hiera_index12 = [[0,3],[3,8],[8,17],[17,20],[20,24]]
hiera12 ={
"hiera_higher":{
        "general_device": [0,3],
        "general_use": [3,8],
        "house_furniture   ": [8,17],
        "room": [17,20],
        "void": [20,24]
   }
}

def prepare_targets(targets):  # targets (b,h,w)
    num_node = targets.shape
    targets_high = torch.ones((num_node), dtype=targets.dtype, device=targets.device)*255 # (40)
    indices_high = []
    for index, high in enumerate(hiera1["hiera_high"].keys()):
        indices = hiera1["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index
        indices_high.append(indices)
    
    return targets, targets_high, indices_high # Tensor:40   Tensor:40

#自己写的三层标签生成 没毛病
def prepare_targets2(targets):  # targets (b,h,w)
    num_node = targets.shape
    targets_high = torch.ones((num_node), dtype=targets.dtype, device=targets.device) * 255  # (40)
    indices_high = []
    for index, high in enumerate(hiera1["hiera_high"].keys()):
        indices = hiera1["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index # 二级类别 # ([ 6, 22, 22, 22, 18,  4,  4,  4,  4, 14, 22,  9,  5, 14, 14, 22,  4, 14]
        indices_high.append(indices)
    targets_higher = torch.ones((num_node), dtype=targets.dtype, device=targets.device) * 255  # (40)
    indices_higher12 = []
    indices_higher2 = []
    for index, high in enumerate(hiera12["hiera_higher"].keys()):  #or hiera2["hiera_higher"]
        indices = hiera12["hiera_higher"][high]                    ## hiera2["hiera_higher"].
        indices2 = hiera2["hiera_higher"][high]
        for ii in range(indices[0], indices[1]):
            targets_higher[targets_high == ii] = index             # [targets == ii]
        indices_higher12.append(indices)
        indices_higher2.append(indices2)
        #  targets_higher ([1, 4, 4, 4, 3, 1, 1, 1, 1, 2, 4, 2, 1, 2, 2, 4, 1, 2]
        #  indices_higher [[0, 3], [3, 7], [7, 16], [16, 19], [19, 23]]
        #  indices_higher2 [[0, 16], [16, 75], [75, 127], [127, 145], [145, 159]]

    return targets, targets_high, targets_higher, indices_high, indices_higher12, indices_higher2  # Tensor:40   Tensor:40

# 自己改写的二层次损失-基本有效
def losses_hiera(predictions, targets, targets_top, num_classes, indices_high, eps=1e-8, gamma=2): #targets ([2, 512, 1024])
    num_node, c = predictions.shape # predictions [9, 184]     Nn, 184

    predictions = torch.sigmoid(predictions.float()) #[9, 184]
    targets = targets.unsqueeze(0) #(1,Nn)
    targets_top = targets_top.unsqueeze(0)
    void_indices = (targets==255) # 没有索引的标签一律归入0类
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,2,1) # 生成onehot gt [1, 160, 9]    bhw -> bchw
    void_indices2 = (targets_top==255)
    targets_top[void_indices2]=0 # 没有二级索引的标签一律归入top 0类
    targets_top = F.one_hot(targets_top, num_classes = 24).permute(0,2,1) # 生成onehot gt [1, 24, 9]

    pred = predictions
    predictions = predictions.permute(1, 0)  # (c ,Nn)
    predictions = predictions.unsqueeze(0)   #([1, 184, 9])

    MCMA = predictions[:,:num_classes,:] #([2, 19, 512, 1024]) 取前160个一级预测
    MCMB = torch.zeros((1,24,num_node), dtype=predictions.dtype, device=predictions.device) # #([1,24,Nn]) 初始化全零的后24个二级预测
    for ii in range(24):
        indices = indices_high[ii]
        MCMB[:,ii:ii+1,:] = torch.max(torch.cat([predictions[:,indices[0]:indices[1],:], predictions[:,num_classes+ii:num_classes+ii+1,:]], dim=1), 1, True)[0]
        #MCMB[:,ii:ii+1,:] = torch.cat([predictions[:,indices_high[ii][0]:indices_high[ii][1],:], predictions[:,num_classes+ii:num_classes+ii+1,:]], dim=1)

    MCLB = predictions[:,num_classes:num_classes+24,:] #取后24个二级预测 #([1,24,Nn])
    MCLA = predictions[:,:num_classes,:].clone() #取前160个一级预测 #([1,160,Nn])
    for ii in range(24):
        indices = indices_high[ii]
        for jj in range(indices[0], indices[1]):
            MCLA[:,jj:jj+1,:] = torch.min(torch.cat([predictions[:,jj:jj+1,:],MCLB[:,ii:ii+1,:]], dim=1), 1, True)[0]
    #print("@@@@@@@Check here")
    loss = ((-targets[:, :num_classes, :] * torch.log(MCLA + eps)- (1.0 - targets[:, :num_classes, :]) * torch.log(1.0 - MCMA + eps))).sum() / num_node / num_classes  # 0.6928   (log_softmax 0.0381)
    loss += ((-targets_top[:, :24, :] * torch.log(MCLB + eps)
              - (1.0 - targets_top[:, :24, :]) * torch.log(1.0 - MCMB + eps))).sum() / num_node / 24  # 1.3950   (log_softmax 0.2613

    return 5*loss

# 自己写的三层次损失-废
def losses_hiera2(predictions, targets, targets_top, targets_top2, num_classes, indices_high,indices_higher12,indices_higher2, eps=1e-8,
                 gamma=2):  # targets ([2, 512, 1024])
    num_node, c = predictions.shape  # predictions ([2, 26, 128, 256])  #([2, 26, 512, 1024])

    predictions = torch.sigmoid(predictions.float())  # ([2, 26, 512, 1024])
    targets = targets.unsqueeze(0)
    targets_top = targets_top.unsqueeze(0)
    targets_top2 = targets_top2.unsqueeze(0)
    void_indices = (targets == 255)  # 没有索引的标签一律归入0类
    targets[void_indices] = 0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 2,1)
    # 生成onehot gt ([2, 512, 1024])->([2, 19, 512, 1024])  (18,) -->(18,160) -->(160,18)
    void_indices2 = (targets_top == 255)
    targets_top[void_indices2] = 0  # 没有二级索引的标签一律归入top 0类
    targets_top = F.one_hot(targets_top, num_classes=24).permute(0, 2,1)  # 生成onehot gt ([2, 512, 1024])--> ([2, 7, 512, 1024])
    targets_top2 = F.one_hot(targets_top2, num_classes=5).permute(0,2,1)

    pred = predictions
    predictions = predictions.permute(1, 0)  # (c , n)
    predictions = predictions.unsqueeze(0)

    MCMA = predictions[:, :num_classes, :]  # ([1, 160, num_node]) 取前160个一级预测
    MCMB = torch.zeros((1, 24, num_node), dtype=predictions.dtype,
                       device=predictions.device)  # #([1, 24,num_node]) 初始化全零的24个二级预测 160~184
    MCMC = torch.zeros((1,5,num_node),dtype=predictions.dtype,
                       device=predictions.device) # (1,5,num_node)
    for ii in range(24):
        indices = indices_high[ii]
        MCMB[:, ii:ii + 1, :] = torch.max(torch.cat([predictions[:, indices_high[ii][0]:indices_high[ii][1], :],
                                        predictions[:, num_classes + ii:num_classes + ii + 1, :]], dim=1),1, True)[0]
        # MCMB concate 底层pred与二层pred


    MCLB = predictions[:, num_classes:num_classes + 24, :]  # 24个二级预测160~184
    MCLC = predictions[:, num_classes+24:,:]
    for iii in range(5):
        MCMC[:,iii:iii + 1,:] = torch.max(torch.cat([predictions[:, indices_higher2[iii][0]:indices_higher2[iii][1],:],
                                                     MCLB[:, indices_higher12[iii][0]:indices_higher12[iii][1],:],
                                                     MCLC[:, iii:iii+1, :],],  dim=1),1, True)[0]


    MCLA = predictions[:, :num_classes, :].clone()  # 取前160个一级预测
    for kk in range(5):
        indices2 = indices_higher12[kk]
        for ii in range(indices_higher12[kk][0], indices_higher12[kk][1]):
            indices = indices_high[ii]
            for jj in range(indices_high[ii][0], indices_high[ii][1]):
                MCLA[:, jj:jj + 1, :] = \
                torch.min(torch.cat([predictions[:, jj:jj + 1, :], MCLB[:, ii:ii + 1, :], MCLC[:,kk:kk+1,:] ], dim=1), 1, True)[0]

    # num_valid  num_valid2 = num_node

    # channel_num*sum()/one_channel_valid already has a weight
    loss = ((-targets[:, :num_classes, :] * torch.log(MCLA + eps)
             - (1.0 - targets[:, :num_classes, :]) * torch.log(1.0 - MCMA + eps))).sum() / num_node / num_classes  # 0.6928   (log_softmax 0.0381)
    loss += ((-targets_top[:, :, :] * torch.log(MCLB + eps)
              - (1.0 - targets_top[:, :, :]) * torch.log(1.0 - MCMB + eps))).sum() / num_node / 24   # 1.3950   (log_softmax 0.2613
    loss += ((-targets_top2[:, :, :] * torch.log(MCLC + eps)
              - (1.0 - targets_top2[:, :, :]) * torch.log(1.0 - MCMC + eps))).sum() / num_node / 5 # 2.1006 (log_softmax 1.3145


    return 5 * loss #  10.5031  (log_softmax 6.5727)


def losses_hiera_focal(predictions, targets, targets_top, num_classes, indices_high, eps=1e-8, gamma=2):
    b, _, h, w = predictions.shape
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets==255)
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    void_indices2 = (targets_top==255)
    targets_top[void_indices2]=0
    targets_top = F.one_hot(targets_top, num_classes = 7).permute(0,3,1,2)

    MCMA = predictions[:,:num_classes,:,:]
    MCMB = torch.zeros((b,7,h, w), dtype=predictions.dtype, device=predictions.device)
    for ii in range(7):
        indices = indices_high[ii]
        MCMB[:,ii:ii+1,:,:] = torch.max(torch.cat([predictions[:,indices_high[ii][0]:indices_high[ii][1],:,:], predictions[:,num_classes+ii:num_classes+ii+1,:,:]], dim=1), 1, True)[0]   
        
    MCLB = predictions[:,num_classes:num_classes+7,:,:]
    MCLA = predictions[:,:num_classes,:,:].clone()
    for ii in range(7):
        indices = indices_high[ii]
        for jj in range(indices_high[ii][0], indices_high[ii][1]):
            MCLA[:,jj:jj+1,:,:] = torch.min(torch.cat([predictions[:,jj:jj+1,:,:],MCLB[:,ii:ii+1,:,:]], dim=1), 1, True)[0]   
            
    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    valid_indices2 = (~void_indices2).unsqueeze(1)
    num_valid2 = valid_indices2.sum()
    #channel_num*sum()/one_channel_valid already has a weight
    loss = ((-targets[:,:num_classes,:,:]*torch.pow((1.0-MCLA),gamma)*torch.log(MCLA+eps)
             -(1.0-targets[:,:num_classes,:,:])*torch.pow(MCMA, gamma)*torch.log(1.0-MCMA+eps))
             *valid_indices).sum()/num_valid/num_classes
    loss+= ((-targets_top[:,:,:,:]*torch.pow((1.0-MCLB), gamma)*torch.log(MCLB+eps)
             -(1.0-targets_top[:,:,:,:])*torch.pow(MCMB, gamma)*torch.log(1.0-MCMB+eps))
             *valid_indices2).sum()/num_valid2/7

    return 5*loss



#@LOSSES.register_module()
class HieraTripletLoss3DSSG(nn.Module):

    def __init__(self,
                 num_classes,
                 use_sigmoid=False,
                 loss_weight=1.0):
        super(HieraTripletLoss3DSSG, self).__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        self.treetripletloss = TreeTripletLoss(160, hiera_map1, hiera_index1)
        self.ce = CrossEntropyLoss()

    def forward(self,
                embedding,
                cls_score,
                label,
                weight=None,
                **kwargs):
        targets, targets_top1,  indices_top1 = prepare_targets(label) #label(Nn,)  targets (Nn) # 由label_map生成high-level的label_map(根据层次索引)
        hiera_loss = losses_hiera(cls_score, targets, targets_top1,  self.num_classes, indices_top1) # 10.5031
        ce_loss = self.ce(cls_score[:, :-24], label)
        ce_loss2 = self.ce(cls_score[:,-24:], targets_top1)
        loss = 0.5 * hiera_loss + ce_loss + ce_loss2

        '''
        #targets, targets_top1, targets_top2, indices_top1,indices_top12,indices_top2 = prepare_targets2(label) #label([2, 512, 1024])  targets ([2, 512, 1024]) # 由label_map生成high-level的label_map(根据层次索引)
        hiera_loss = losses_hiera2(cls_score, targets, targets_top1, targets_top2, self.num_classes, indices_top1, indices_top12, indices_top2)
        ce_loss = self.ce(cls_score[:, :-29], label) #5.0741
        ce_loss2 = self.ce(cls_score[:,-29:-5],targets_top1)# 3.1615
        ce_loss3 = self.ce(cls_score[:,:-5],targets_top2) # 5.2122 ##############################
        loss = 0.5*hiera_loss + ce_loss + ce_loss2 #+ ce_loss3
        '''

        embedding = embedding.unsqueeze(0).permute(0,2,1) #(b, 256,Nn)
        label = label.unsqueeze(0) #(b,Nn)
        loss_triplet, class_count = self.treetripletloss(embedding, label)
        loss +=  loss_triplet

        #class_count = [torch.ones_like(class_count)] # for _ in range(torch.distributed.get_world_size())]###???
        #torch.distributed.all_gather(class_counts, class_count, async_op=False)
        #class_counts = torch.cat(class_counts, dim=0)
        #if torch.distributed.get_world_size()==torch.nonzero(class_counts, as_tuple=False).size(0):
        #factor = 1/4*(1+torch.cos(torch.tensor((step-80000)/80000*math.pi))) if step<80000 else 0.5 # ????
        #loss+=factor*loss_triplet
        '''print("######### Step: ", step)
        print("######## loss:", loss)
        print("######## loss_triplet:", loss_triplet)
        print("##########class-count:\n", class_count)'''

        return loss#*self.loss_weight

