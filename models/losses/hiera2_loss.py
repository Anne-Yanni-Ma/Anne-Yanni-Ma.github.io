import torch
import torch.nn as nn
import torch.nn.functional as F


def losses_hiera2(predictions, targets, targets_middle, targets_top, num_classes, indices_middle, indices_mt, eps=1e-8, gamma=2):
    num_node, c = predictions.shape  #
    pred = predictions
    predictions = predictions.permute(1, 0)  # (c , n)
    predictions = predictions.unsqueeze(0)
    predictions = torch.sigmoid(predictions.float())
    targets = targets.unsqueeze(0)
    targets_middle = targets_middle.unsqueeze(0)
    targets_top = targets_top.unsqueeze(0)
    void_indices = (targets==255)
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0, 2,1)
    targets_middle = F.one_hot(targets_middle, num_classes=24).permute(0, 2,1)  # 生成onehot gt ([2, 512, 1024])--> ([2, 7, 512, 1024])
    targets_top = F.one_hot(targets_top, num_classes=5).permute(0,2,1)

    MCMA = predictions[:,:num_classes,:] #前160个预测
    MCMB = torch.zeros((1, 24, num_node), dtype=predictions.dtype,
                       device=predictions.device)  # #([1, 24,num_node]) 初始化全零的24个二级预测 160~184
    MCMC = torch.zeros((1, 5, num_node), dtype=predictions.dtype,
                       device=predictions.device)  # (1,5,num_node)

    for ii in range(24):
        indices = indices_middle[ii]
        MCMB[:, ii:ii + 1, :] = torch.max(torch.cat([predictions[:, indices_middle[ii][0]:indices_middle[ii][1], :],
                                        predictions[:, num_classes + ii:num_classes + ii + 1, :]], dim=1),1, True)[0]
        # MCMB concate 底层pred与二层pred  #第二层类的底３层预测1和二层预测对应位置concat　　占nc+24
    for ii in range(5):
        MCMC[:, ii:ii + 1, :] = torch.max(torch.cat([MCMB[:, indices_mt[ii][0]:indices_mt[ii][1],:],
                                        predictions[:, num_classes+24+ii : num_classes+24+ii+1,:]], dim=1),1, True)[0]

    # ---------------------------------------------
    '''MCMB_back = torch.max(torch.cat([predictions[:,0:1,:],predictions[:,num_classes:num_classes+1,:]], dim=1), 1, True)[0] # 背景预测 第一层类　0和nc+1
    MCMB1 = torch.max(torch.cat([predictions[:,ii:ii+1,:] for ii in upper_ids]+[predictions[:,num_classes+1:num_classes+2,:]], dim=1), 1, True)[0]#第二层类的底３层预测1和二层预测对应位置concat　　占nc+2
    MCMB2 = torch.max(torch.cat([predictions[:,ii:ii+1,:] for ii in lower_ids]+[predictions[:,num_classes+2:num_classes+3,:]], dim=1), 1, True)[0] #第二层类的底３层预测2和二层预测对应位置concat　　占nc+3
    MCMB = torch.cat([MCMB_back, MCMB1, MCMB2], dim=1)  # concat 第二层类的预测
    MCMC_back = torch.max(torch.cat([MCMB_back ,predictions[:,num_classes+3:num_classes+4,:]], dim=1), 1, True)[0] #背景预测对应位置　占nc+4
    MCMC1 = torch.max(torch.cat([MCMB1, MCMB2, predictions[:,num_classes+4:num_classes+5,:]], dim=1), 1, True)[0]  # 二层预测对应位置　占nc+5
    MCMC = torch.cat([MCMC_back, MCMC1], dim=1)    #第一层和第二层结合'''
   # ---------------------------------------------


    MCLC = predictions[:,num_classes+24:,:] # [:,5,:]
    #MCLC = predictions[:, num_classes+3:num_classes+5, :]  #
    MCLB = torch.zeros((1, 24, num_node), dtype=predictions.dtype,
                       device=predictions.device)  # #([1, 24,num_node]) 初始化全零的24个二级预测 160~184
    MCLA = torch.zeros((1, 24, num_node), dtype=predictions.dtype,
                       device=predictions.device)  # (1,5,num_node)

    '''MCLB_back = torch.min(torch.cat([MCLC[:,0:1,:], predictions[:,num_classes:num_classes+1,:]], dim=1), 1, True)[0]
    MCLB1 = torch.min(torch.cat([MCLC[:,1:2,:], predictions[:,num_classes+1:num_classes+2,:]], dim=1), 1, True)[0]  # 自顶向下对应　
    MCLB2 = torch.min(torch.cat([MCLC[:,1:2,:], predictions[:,num_classes+2:num_classes+3,:]], dim=1), 1, True)[0]
    MCLB = torch.cat((MCLB_back, MCLB1, MCLB2), dim=1)
    MCLA_back = torch.min(torch.cat([predictions[:, 0:1, :], MCLB[:, 0:1, :]], dim=1), 1, True)[0]
    MCLA1 = torch.cat(
        [torch.min(torch.cat([predictions[:, ii:ii + 1, :], MCLB[:, 1:2, :]], dim=1), 1, True)[0] for ii in upper_ids],
        dim=1)
    MCLA2 = torch.cat(
        [torch.min(torch.cat([predictions[:, ii:ii + 1, :], MCLB[:, 2:3, :]], dim=1), 1, True)[0] for ii in lower_ids],
        dim=1)
    if len(upper_ids) > 5:
        MCLA = torch.cat(
            [MCLA_back, MCLA1[:, 0:7, :], MCLA2[:, 0:2, :], MCLA1[:, 7:9, :], MCLA2[:, 2:3, :], MCLA1[:, 9:12, :],
             MCLA2[:, 3:7, :]], dim=1)
    else:
        MCLA = torch.cat([MCLA_back, MCLA1, MCLA2], dim=1)'''

    for kk in range(5):
        for ii in range(indices_mt[kk][0],indices_mt[kk][1]):
            MCLB[:, ii:ii+1,:] = torch.min(torch.cat([MCLC[:,kk:kk+1,:],  predictions[:, num_classes+ii :num_classes+ii+1,:]], dim=1),1, True)[0]  #[:,24,:]

    for jj in range(24):
        for ii in range(indices_middle[jj][0], indices_middle[jj][1]):
            MCLA[:,jj:jj+1,:] = torch.min(torch.cat([predictions[:,ii:ii+1,:], MCLB[:,jj:jj+1,:]], dim=1),1, True)[0] #[:,24,:]


    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    loss = ((-targets[:,:num_classes,:]*torch.log(MCLA+eps)
            -(1-targets[:,:num_classes,:])*torch.log(1-MCMA+eps)) ##################
            *valid_indices).sum()/num_valid/num_classes
    loss+= ((-targets_middle[:,:,:]*torch.log(MCLB+eps)
            -(1-targets_middle[:,:,:])*torch.log(1-MCMB+eps))
            *valid_indices).sum()/num_valid/24
    loss+= ((-targets_top[:,:,:]*torch.log(MCLC+eps)
            -(1-targets_top[:,:,:])*torch.log(1-MCMC+eps))
            *valid_indices).sum()/num_valid/5

    return 5*loss

def losses_hiera2_focal(predictions, targets, targets_middle, targets_top, num_classes, upper_ids, lower_ids, eps=1e-8, gamma=2):
    predictions = torch.sigmoid(predictions.float())
    void_indices = (targets==255)
    targets[void_indices]=0
    targets = F.one_hot(targets, num_classes=num_classes).permute(0,3,1,2)
    targets_middle = F.one_hot(targets_middle, num_classes = 3).permute(0,3,1,2)
    targets_top = F.one_hot(targets_top, num_classes = 2).permute(0,3,1,2)

    MCMA = predictions[:,:num_classes,:,:]
    MCMB_back = torch.max(torch.cat([predictions[:,0:1,:,:],predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCMB1 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in upper_ids]+[predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCMB2 = torch.max(torch.cat([predictions[:,ii:ii+1,:,:] for ii in lower_ids]+[predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCMB = torch.cat([MCMB_back, MCMB1, MCMB2], dim=1)
    MCMC_back = torch.max(torch.cat([MCMB_back ,predictions[:,num_classes+3:num_classes+4,:,:]], dim=1), 1, True)[0]
    MCMC1 = torch.max(torch.cat([MCMB1, MCMB2, predictions[:,num_classes+4:num_classes+5,:,:]], dim=1), 1, True)[0]
    MCMC = torch.cat([MCMC_back, MCMC1], dim=1)
    

    MCLC = predictions[:,num_classes+3:num_classes+5,:,:]
    MCLB_back = torch.min(torch.cat([MCLC[:,0:1,:,:], predictions[:,num_classes:num_classes+1,:,:]], dim=1), 1, True)[0]
    MCLB1 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+1:num_classes+2,:,:]], dim=1), 1, True)[0]
    MCLB2 = torch.min(torch.cat([MCLC[:,1:2,:,:], predictions[:,num_classes+2:num_classes+3,:,:]], dim=1), 1, True)[0]
    MCLB = torch.cat((MCLB_back, MCLB1, MCLB2), dim=1)
    MCLA_back = torch.min(torch.cat([predictions[:,0:1,:,:],MCLB[:,0:1,:,:]], dim=1), 1, True)[0]
    MCLA1 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,1:2,:,:]], dim=1), 1, True)[0] for ii in upper_ids], dim=1)
    MCLA2 = torch.cat([torch.min(torch.cat([predictions[:,ii:ii+1,:,:],MCLB[:,2:3,:,:]], dim=1), 1, True)[0] for ii in lower_ids], dim=1)
    if len(upper_ids)>5:
        MCLA = torch.cat([MCLA_back, MCLA1[:,0:7,:,:], MCLA2[:,0:2,:,:], MCLA1[:,7:9,:,:], MCLA2[:,2:3,:,:], MCLA1[:,9:12,:,:], MCLA2[:,3:7,:,:]], dim=1)
    else:
        MCLA = torch.cat([MCLA_back, MCLA1, MCLA2], dim=1)

    valid_indices = (~void_indices).unsqueeze(1)
    num_valid = valid_indices.sum()
    loss = ((-targets[:,:num_classes,:,:]*torch.pow((1.0-MCLA),gamma)*torch.log(MCLA+eps)
             -(1.0-targets[:,:num_classes,:,:])*torch.pow(MCMA, gamma)*torch.log(1.0-MCMA+eps))
             *valid_indices).sum()/num_valid/num_classes
    loss+= ((-targets_middle*torch.pow((1.0-MCLB), gamma)*torch.log(MCLB+eps)
             -(1.0-targets_middle)*torch.pow(MCMB, gamma)*torch.log(1-MCMB+eps))
             *valid_indices).sum()/num_valid/3
    loss+= ((-targets_top*torch.pow((1-MCLC), gamma)*torch.log(MCLC+eps)
             -(1.0-targets_top)*torch.pow(MCMC, gamma)*torch.log(1-MCMC+eps))
             *valid_indices).sum()/num_valid/2

    return loss
