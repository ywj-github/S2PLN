from utils.utils import AverageMeter, accuracy
from utils.statistic import get_EER_states, get_HTER_at_thr, calculate, calculate_threshold,get_EER_states_I
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def eval3(valid_dataloader, G_fas, Cls_fas, Dec_fas, norm_flag):
    criterion = nn.CrossEntropyLoss()
    valid_losses = AverageMeter()
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    G_fas.eval()
    Dec_fas.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(valid_dataloader):
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()

            feature_fas, outs = G_fas(input)
            cls_out = Cls_fas(feature_fas, norm_flag)
            decoder_fas = Dec_fas(outs)[-1]
            # decoder_fas = (Dec_fas(outs)[-1] + 1)/2

            spoof_score = torch.mean(torch.abs(decoder_fas).view(decoder_fas.shape[0], -1), dim=1)
            spoof_score = spoof_score.cpu().data.numpy()
            label = 1 - target.cpu().data.numpy()
            videoID = videoID.cpu().data.numpy()
            for i in range(len(spoof_score)):
                if(videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(spoof_score[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(spoof_score[i])
                    label_dict[videoID[i]].append(label[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 2))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
    prob_list = []
    label_list = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = (sum(target_dict_tmp[key]) / len(target_dict_tmp[key])).long()
        loss = criterion(avg_single_video_output, avg_single_video_target)
        acc_valid = accuracy(avg_single_video_output, avg_single_video_target, topk=(1,))
        valid_losses.update(loss.item())
        valid_top1.update(acc_valid[0])
    auc_score = roc_auc_score(label_list, prob_list)
    cur_EER_valid, threshold, _, _ = get_EER_states(prob_list, label_list)
    ACC_threshold, APCER, BPCER = calculate_threshold(prob_list, label_list, threshold)
    cur_HTER_valid = get_HTER_at_thr(prob_list, label_list, threshold)

    return [valid_losses.avg, valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, threshold, ACC_threshold*100, APCER*100, BPCER*100]