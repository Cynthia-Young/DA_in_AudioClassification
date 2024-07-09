import argparse
import os
from os import path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
from torch.utils.data import DataLoader
import random
from scipy.spatial.distance import cdist
from data_load import audio_dataset_read
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import do_mixup


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True, size_average=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.size_average = size_average
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        if self.size_average:
            loss = (- targets * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).sum(1)
        return loss


class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []

        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)


def digit_load(args): 
    source_bs = args.batch_size
    target_bs = args.mozilla_batch_size

    s_domain = args.dset[0]
    t_domain = args.dset[2]

    train_source, test_source = audio_dataset_read(args, s_domain)
    train_target, _ = audio_dataset_read(args, t_domain, index=True)
    _, test_target = audio_dataset_read(args, t_domain)

    dset_loaders = {}
    dset_loaders["source_tr"] = DataLoader(train_source, 
        batch_size = source_bs * 2 if 'mixup' in args.augmentation else source_bs,
        shuffle=True, num_workers=args.worker, drop_last=False)
    dset_loaders["source_te"] = DataLoader(test_source, batch_size=source_bs*2, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target"] = DataLoader(train_target, batch_size=target_bs, shuffle=True, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["target_te"] = DataLoader(train_target, batch_size=target_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    dset_loaders["test"] = DataLoader(test_target, batch_size=target_bs, shuffle=False, 
        num_workers=args.worker, drop_last=False)
    return dset_loaders

def cal_acc(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, embedding = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    _, all_label = torch.max(all_label, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()
    return accuracy*100, mean_ent


def train_source(args):
    dset_loaders = digit_load(args)
    
    ## set base network
    model = network.Transfer_Cnn14(sample_rate = 32000, window_size = 1024, hop_size= 320, mel_bins = 64, fmin = 50, fmax = 14000, 
        classes_num = 3, freeze_base = args.freeze_base, freeze_classifier = False).cuda()
    pretrain = True if args.pretrained_checkpoint_path else False
    if pretrain:
        model.load_from_pretrain(args.pretrained_checkpoint_path)

    param_group = []
    learning_rate = args.lr
    for k, v in model.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    if 'mixup' in args.augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)

    acc_init = 0
    max_iter = args.max_epoch * len(dset_loaders["source_tr"])
    interval_iter = max_iter // 100
    iter_num = 0

    acc_list_tr = []  # 用于记录训练集准确率
    acc_list_te = []  # 用于记录测试集准确率

    model.train()

    with tqdm(total=max_iter) as pbar:
        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source) #yxy
            except:
                iter_source = iter(dset_loaders["source_tr"])
                inputs_source, labels_source = next(iter_source) #yxy

            if inputs_source.size(0) % 2 == 1:
                continue

            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
            
            # 得到用于mixup的比例
            if 'mixup' in args.augmentation:
                mixup_lambda = torch.tensor(mixup_augmenter.get_lambda(len(inputs_source))).cuda()
            else:
                mixup_lambda = None
        
            # 得到最终预测结果和标签 (aug or not)
            if 'mixup' in args.augmentation:
                clipwise_output, embedding = model(inputs_source, mixup_lambda.float())
                labels_aug = do_mixup(labels_source, mixup_lambda)
            else:
                clipwise_output, embedding = model(inputs_source, None)
                labels_aug = labels_source

            classifier_loss = CrossEntropyLabelSmooth(num_classes=args.class_num, epsilon=args.smooth)(clipwise_output, labels_aug)            
            optimizer.zero_grad()
            classifier_loss.backward()
            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                model.eval()
                acc_s_tr, _ = cal_acc(dset_loaders['source_tr'], model)
                acc_s_te, _ = cal_acc(dset_loaders['source_te'], model)
                acc_list_tr.append(acc_s_tr)
                acc_list_te.append(acc_s_te)
                log_str = 'Train Source Task: {}, Iter:{}/{}; Accuracy = {:.2f}%/ {:.2f}%'.format(args.dset, iter_num, max_iter, acc_s_tr, acc_s_te)
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str+'\n')

                if acc_s_te >= acc_init:
                    acc_init = acc_s_te
                    best_model = model.state_dict()

                model.train()

            pbar.update(1)
    
     # 绘制准确率随着迭代次数的变化图
    plt.figure()
    plt.plot(range(1, len(acc_list_tr)*400 + 1, 400), acc_list_tr, label='Train Accuracy')
    plt.plot(range(1, len(acc_list_te)*400 + 1, 400), acc_list_te, label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Iterations')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_source_s2m_32_100_panns_accuracy_400_0.01_newdspre.png')  # 保存图像
    plt.show()

    torch.save(best_model, osp.join(args.output_dir, "source_panns.pt"))

    return model

def test_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    # netF = network.AudioClassifier().cuda()
    # netC = network.feat_classifier(type=args.layer, class_num = args.class_num, bottleneck_dim=args.bottleneck).cuda()

    netG = network.Transfer_Cnn14(sample_rate = 32000, window_size = 1024, hop_size= 320, mel_bins = 64, fmin = 50, fmax = 14000, 
        classes_num = 3, freeze_base = False, freeze_classifier = False).cuda()
    model = nn.Sequential(netG)
    args.modelpath = args.output_dir + '/ps_0.0_par_0.3final.pt'   
    model.load_state_dict(torch.load(args.modelpath))
    model.eval()

    acc, _ = cal_acc(dset_loaders['test'], model)
    log_str = 'Test Target Task: {}, Accuracy = {:.2f}%'.format(args.dset, acc)
    print(log_str+'\n')

def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s

def train_target(args):
    dset_loaders = digit_load(args)
    ## set base network
    model = network.Transfer_Cnn14(sample_rate = 32000, window_size = 1024, hop_size= 320, mel_bins = 64, fmin = 50, fmax = 14000, 
        classes_num = 3, freeze_base = args.freeze_base, freeze_classifier = True).cuda()
    
    args.modelpath = args.output_dir + '/source_panns.pt'   
    model.load_state_dict(torch.load(args.modelpath))

    param_group = []
    for k, v in model.base.named_parameters():
        param_group += [{'params': v, 'lr': args.lr}]
    
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = len(dset_loaders["target"])
    iter_num = 0

    with tqdm(total=max_iter) as pbar:
        while iter_num < max_iter:
            optimizer.zero_grad()
            try:
                inputs_test, _, tar_idx = next(iter_test)
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, _, tar_idx = next(iter_test)

            if inputs_test.size(0) == 1:
                continue

            if iter_num % interval_iter == 0 and args.cls_par > 0:
                model.eval()
                mem_label = obtain_label(dset_loaders['target_te'], model, args)
                mem_label = torch.from_numpy(mem_label).cuda()
                model.train()

            iter_num += 1
            lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)

            inputs_test = inputs_test.cuda()
            outputs_test, features_test = model(inputs_test)

            if args.cls_par > 0:
                pred = mem_label[tar_idx]
                classifier_loss = args.cls_par * nn.CrossEntropyLoss()(outputs_test, pred)
            else:
                classifier_loss = torch.tensor(0.0).cuda()

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(Entropy(softmax_out))
                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                    entropy_loss -= torch.sum(-msoftmax * torch.log(msoftmax + 1e-5))

                im_loss = entropy_loss * args.ent_par
                classifier_loss += im_loss

            optimizer.zero_grad()
            classifier_loss.backward()

            optimizer.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                model.eval()
                acc, _ = cal_acc(dset_loaders['test'], model)
                log_str = 'Train Target Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.dset, iter_num, max_iter, acc)
                args.out_file.write(log_str + '\n')
                args.out_file.flush()
                print(log_str+'\n')
                model.train()
            pbar.update(1)

    if args.issave:
        torch.save(model.state_dict(), osp.join(args.output_dir, "target_panns_" + args.savename + ".pt"))
        
    return model

def obtain_label(loader, model, args, c=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs, feas = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    _, all_label = torch.max(all_label, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy*100, acc*100)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str+'\n')
    return pred_label.astype('int')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT++')
    parser.add_argument('--dset', type=str, default='m2s')
    
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--max_epoch', type=int, default=50, help="maximum epoch") #50
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--mozilla_batch_size', type=int, default=32, help="mozilla_batch_size")
    parser.add_argument('--worker', type=int, default=16, help="number of workers")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")

    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--ent_par', type=float, default=1.0)
    
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)

    parser.add_argument('--bottleneck', type=int, default=9)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--smooth', type=float, default=0.1)   
    parser.add_argument('--ssl', type=float, default=0) 
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--issave', type=bool, default=True)

    parser.add_argument('--freeze_base', action='store_true', default=False)
    parser.add_argument('--augmentation', type=str, choices=['none', 'mixup'], default='mixup')

    args = parser.parse_args()
    args.class_num = 3
    args.pretrained_checkpoint_path = "code/Cnn14_mAP=0.431.pth"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    args.output_dir = osp.join(args.output, 'seed' + str(args.seed), args.dset)
    if not osp.exists(args.output_dir):
        os.system('mkdir -p ' + args.output_dir)
    if not osp.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if not osp.exists(osp.join(args.output_dir + '/source_panns.pt')):
        args.out_file = open(osp.join(args.output_dir, 'log_src.txt'), 'w')
        args.out_file.write(print_args(args)+'\n')
        args.out_file.flush()
        train_source(args)
        test_target(args)

    test_target(args)