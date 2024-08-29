import os
import sys
import argparse
import logging
import time
import datetime
import importlib
import shutil
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
#from data.dataloader import DataLoader_3DSSG
from data.dataloader_EW import DataLoader_3DSSG

from ssg_eval_tool import Object_Accuracy, Object_Recall, Predicate_Accuracy, Predicate_Recall, Relation_Recall


# Project directory
PROJECT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(PROJECT_DIR, 'models'))


# Arguments declearation
def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='GNN_nofusion', help='model name [default: pointnet2,GNN_knowledge_fusion]')
    parser.add_argument('--epoch',  default=100, type=int, help='Epoch to run [default: 100]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float,  default=0.7, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


def prepare_onehot_objgt(self, gt_obj):
    insnum = gt_obj.shape[0]
    onehot = torch.zeros(insnum, 160).float().cuda()
    for i in range(insnum):
        onehot[i, gt_obj[i]] = 1
    return onehot


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # cuda
    # torch.backends.cudnn.enabled = False  # disable cudnn for CUDNN_NOT_SUPPORT, may not contiguous
    # ---------------- Create log dir -------------------
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('GNN_SGCls')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # ---------------- Training set --------------------
    TRAINING_SET = DataLoader_3DSSG(training=True)
    TEST_SET = DataLoader_3DSSG(training=False)
    trainDataLoader = torch.utils.data.DataLoader(TRAINING_SET, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_SET, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn=lambda x: np.random.seed(x+int(time.time())))
    # ---------------- Log dataset info --------------------
    log_string("The number of training data is: %d" % len(TRAINING_SET))

    # ---------------- Config network model --------------------
    MODEL = importlib.import_module(args.model)
    shutil.copy(os.path.join(PROJECT_DIR, 'models/%s.py' % args.model), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/pointnet.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/gnn_models.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/graph.py'), str(experiment_dir))
    shutil.copy(os.path.join(PROJECT_DIR, 'models/utils.py'), str(experiment_dir))

    # initial pretrained node/edge embedder
    address_node = os.path.join(PROJECT_DIR, 'log/obj_classification/2023-03-05_20-52')
    sys.path.append(address_node)
    MODEL_node = importlib.import_module('models.obj_classification.obj_classification')
    node_embedder = MODEL_node.get_model().cuda()
    dict_address_node = os.path.join(address_node, 'checkpoints/last_model.pth')
    checkpoint_node = torch.load(dict_address_node)
    node_embedder.load_state_dict(checkpoint_node['model_state_dict'])
    node_embedder = node_embedder.cuda().eval()

    address_edge = os.path.join(PROJECT_DIR, 'log/pred_classification/2023-03-05_20-53')
    sys.path.append(address_edge)
    MODEL_edge = importlib.import_module('models.pred_classification.pred_classification')
    edge_embedder = MODEL_edge.get_model().cuda()
    dict_address_edge = os.path.join(address_edge, 'checkpoints/last_model.pth')
    checkpoint_edge = torch.load(dict_address_edge)
    edge_embedder.load_state_dict(checkpoint_edge['model_state_dict'])
    edge_embedder = edge_embedder.cuda().eval()

    obj_w = TRAINING_SET.obj_w
    pred_w = TRAINING_SET.pred_w
    network = MODEL.get_model().cuda()    # cuda
    criterion = MODEL.get_loss(alpha=0.1, beta=1, gamma=2, obj_w=obj_w, pred_w=pred_w).cuda()   # cuda

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Conv1d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    start_epoch = 0
    network = network.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(network.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-8
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_loss = 999
    best_rel_R20 = 0
    best_rel_R50 = 0
    best_rel_R100 = 0
    best_rel_R20_ngc = 0
    best_rel_R50_ngc = 0
    best_rel_R100_ngc = 0
    best_rel_mR20 = 0
    best_rel_mR50 = 0
    best_rel_mR100 = 0

    # ---------------- Start training --------------------
    for epoch in range(start_epoch, args.epoch):
        '''Train on scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        network = network.apply(lambda x: bn_momentum_adjust(x, momentum))

        loss_sum = 0
        train_obj_acc = Object_Accuracy(len(trainDataLoader), need_softmax=False)
        train_obj_recall = Object_Recall(len(trainDataLoader), need_softmax=False)
        train_pred_acc = Predicate_Accuracy(len(trainDataLoader), need_softmax=False)
        train_pred_recall = Predicate_Recall(len(trainDataLoader), need_softmax=False)
        train_rel_recall = Relation_Recall(len(trainDataLoader), need_softmax=False)
        # ---------------- Start batch set training --------------------
        bar = tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9)
        for i, data in bar:
            pc_mat, gt_obj, gt_rel = data
            pc_mat = pc_mat[0]
            gt_obj = gt_obj[0]
            gt_rel = gt_rel[0]

            pc_mat_node = pc_mat.cuda()
            pc_mat_edge = torch.clone(pc_mat_node).cuda()
            gt_obj = gt_obj.cuda().long()
            gt_rel = gt_rel.cuda().long()

            _, obj_codes = node_embedder(pc_mat_node)  #([7, 1024, 3]) #([7, 512])
            _, pred_codes = edge_embedder(pc_mat_edge)

            optimizer.zero_grad()
            network = network.train()
            node_output, edge_output = network(obj_codes, pred_codes)
            loss = criterion(node_output, edge_output, gt_obj, gt_rel)
            loss.backward()
            optimizer.step()

            loss_sum = loss_sum + loss.item()
            node_output_eval = node_output.clone().detach()
            edge_output_eval = edge_output.clone().detach()

            train_obj_acc.calculate_accuray(node_output_eval, gt_obj)
            train_obj_recall.calculate_recall(node_output_eval, gt_obj)
            train_pred_acc.calculate_accuracy_binary(pc_mat.shape[0], edge_output_eval, gt_rel) #　对有无关系的预测  edge_output_eval(Ne,27)
            train_pred_acc.calculate_recall_binary(pc_mat.shape[0], edge_output_eval, gt_rel)
            train_pred_acc.calculate_accuracy(pc_mat.shape[0], edge_output_eval, gt_rel)
            train_pred_recall.calculate_recall(pc_mat.shape[0], edge_output_eval, gt_rel)
            train_rel_recall.calculate_recall(node_output_eval, edge_output_eval, gt_obj, gt_rel)
            train_rel_recall.calculate_ngc_recall(node_output_eval, edge_output_eval, gt_obj, gt_rel)
        train_obj_acc.final_update()
        train_obj_recall.final_update()
        train_pred_acc.final_update()
        train_pred_recall.final_update()
        train_rel_recall.final_update()
        log_string('Training mean loss: %f' % (loss_sum / len(trainDataLoader)))
        log_string('Training ' + train_obj_acc.print_string())
        log_string('Training ' + train_obj_recall.print_string())
        log_string('Training ' + train_pred_acc.print_string())
        log_string('Training ' + train_pred_recall.print_string())
        train_recall_res, train_recall_res_ngc, train_m_recall_res = train_rel_recall.print_string()
        log_string('Training ' + train_recall_res)
        log_string('Training NGC ' + train_recall_res_ngc)
        log_string('Training mean ' + train_m_recall_res)

        test_obj_acc = Object_Accuracy(len(testDataLoader), need_softmax=False)
        test_obj_recall = Object_Recall(len(testDataLoader), need_softmax=False)
        test_pred_acc = Predicate_Accuracy(len(testDataLoader), need_softmax=False)
        test_pred_recall = Predicate_Recall(len(testDataLoader), need_softmax=False)
        test_rel_recall = Relation_Recall(len(testDataLoader), need_softmax=False)
        with torch.no_grad():
            loss_sum = 0
            log_string('---- EPOCH %03d TEST ----' % (global_epoch + 1))
            test_bar = tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9)
            for i, data in test_bar:
                pc_mat, gt_obj, gt_rel = data
                pc_mat = pc_mat[0]
                gt_obj = gt_obj[0]
                gt_rel = gt_rel[0]

                pc_mat_node = pc_mat.cuda()
                pc_mat_edge = torch.clone(pc_mat_node).cuda()
                gt_obj = gt_obj.cuda().long()
                gt_rel = gt_rel.cuda().long()

                node_embedder = node_embedder.eval()
                edge_embedder = edge_embedder.eval()

                _, obj_codes = node_embedder(pc_mat_node)
                _, pred_codes = edge_embedder(pc_mat_edge)

                network = network.eval()
                node_output, edge_output = network(obj_codes, pred_codes)
                loss = criterion(node_output, edge_output, gt_obj, gt_rel)

                loss_sum = loss_sum + loss.item()
                node_output_eval = node_output.clone().detach()
                edge_output_eval = edge_output.clone().detach()
                test_obj_acc.calculate_accuray(node_output_eval, gt_obj)
                test_obj_recall.calculate_recall(node_output_eval, gt_obj)
                test_pred_acc.calculate_accuracy_binary(pc_mat.shape[0], edge_output_eval, gt_rel)
                test_pred_acc.calculate_recall_binary(pc_mat.shape[0], edge_output_eval, gt_rel)
                test_pred_acc.calculate_accuracy(pc_mat.shape[0], edge_output_eval, gt_rel)
                test_pred_recall.calculate_recall(pc_mat.shape[0], edge_output_eval, gt_rel)
                test_rel_recall.calculate_recall(node_output_eval, edge_output_eval, gt_obj, gt_rel)
                test_rel_recall.calculate_ngc_recall(node_output_eval, edge_output_eval, gt_obj, gt_rel)
            test_obj_acc.final_update()
            test_obj_recall.final_update()
            test_pred_acc.final_update()
            test_pred_recall.final_update()
            test_rel_recall.final_update()
            log_string('Evaluation mean loss: %f' % (loss_sum / len(testDataLoader)))
            log_string('Eval ' + test_obj_acc.print_string())
            log_string('Eval ' + test_obj_recall.print_string())
            log_string('Eval ' + test_pred_acc.print_string())
            log_string('Eval ' + test_pred_recall.print_string())
            test_recall_res, test_recall_res_ngc, test_m_recall_res = test_rel_recall.print_string()
            log_string('Eval ' + test_recall_res)
            log_string('Eval NGC ' + test_recall_res_ngc)
            log_string('Eval mean ' + test_m_recall_res)

            curr_loss = loss_sum / len(testDataLoader)
            curr_rel_R20 = test_rel_recall.recall[20]
            curr_rel_R50 = test_rel_recall.recall[50]
            curr_rel_R100 = test_rel_recall.recall[100]
            curr_rel_R20_ngc = test_rel_recall.ngc_recall[20]
            curr_rel_R50_ngc = test_rel_recall.ngc_recall[50]
            curr_rel_R100_ngc = test_rel_recall.ngc_recall[100]
            curr_rel_mR20 = test_rel_recall.m_recall[20]
            curr_rel_mR50 = test_rel_recall.m_recall[50]
            curr_rel_mR100 = test_rel_recall.m_recall[100]

            if curr_rel_R50 >best_rel_R50 or curr_rel_R100 >best_rel_R100:
                best_rel_R20 = curr_rel_R20
                best_rel_R50 = curr_rel_R50
                best_rel_R100 = curr_rel_R100
                log_string('--Eval-- Best rel: R@20: %f; '%(best_rel_R20) + 'rel R@50: %f; '%(best_rel_R50)+'rel R@100: %f; '% (best_rel_R100))

            if curr_rel_R50_ngc >best_rel_R50_ngc or curr_rel_R100_ngc >best_rel_R100_ngc:
                best_rel_R20_ngc = curr_rel_R20_ngc
                best_rel_R50_ngc = curr_rel_R50_ngc
                best_rel_R100_ngc = curr_rel_R100_ngc
                log_string('--Eval-- Best NGC rel: R@20: %f; '%(best_rel_R20_ngc) + 'rel R@50: %f; '%(best_rel_R50_ngc)+'rel R@100: %f; '% (best_rel_R100_ngc))
                logger.info('Save NGC model...')
                savepath = str(checkpoints_dir) + '/best_NGCrel_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'current_loss': curr_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            if curr_rel_mR50 > best_rel_mR50 or curr_rel_mR100 > best_rel_mR100:
                '''best_rel_R20 = curr_rel_R20
                best_rel_R50 = curr_rel_R50
                best_rel_R100 = curr_rel_R100
                best_rel_R20_ngc = curr_rel_R20_ngc
                best_rel_R50_ngc = curr_rel_R50_ngc
                best_rel_R100_ngc = curr_rel_R100_ngc'''
                best_rel_mR20 = curr_rel_mR20
                best_rel_mR50 = curr_rel_mR50
                best_rel_mR100 = curr_rel_mR100
                best_loss = curr_loss
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_mR_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'current_loss': curr_loss,
                    'model_state_dict': network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')

            log_string('Best mean loss: %f' % (best_loss))
            log_string('Curr rel R@20: %f; ' % (curr_rel_R20) + 'rel R@50: %f; ' % (curr_rel_R50) + 'rel R@100: %f;' % (curr_rel_R100))
            log_string('Curr NGC rel R@20: %f; ' % (curr_rel_R20_ngc) + 'rel R@50: %f; ' % (curr_rel_R50_ngc) + 'rel R@100: %f;' % (curr_rel_R100_ngc))
            log_string('--Eval-- Best rel mean R@20: %f; ' % (best_rel_mR20) + 'rel R@50: %f; ' % (best_rel_mR50) + 'rel R@100: %f;' % (best_rel_mR100))
        global_epoch += 1



if __name__ == "__main__":
    args = parse_args()
    main(args)
