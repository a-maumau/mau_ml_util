"""
  input should be a torch.Tensor
  it would be easy to implement in numpy, I think.
"""
import torch
import torch.nn as nn
import numpy as np

eps = 1e-8

def pixel_accuracy(pred_labels, gt_labels, class_num=2, class_acc=False, size_average=True):
    """Return the pixel accuracy        
        args
            pred_labels and gt_labels should be batch x w x h

        return
            mean pixel accuracy.
            if the size_mean is False, it returns a vector of pixel accuracy
    """
    if class_acc:
        """
            acc is going like
            acc = {"class_0":0.39, "class_1":0.114, "class":0.514, ..., ",mean_all_pixel":0.810}
        """
        acc = {}
        batch_size = pred_labels.shape[0]

        for cls_id in range(class_num):
            batch_result = []

            class_mask = (gt_labels==cls_id).view(batch_size, -1).type(torch.FloatTensor) # 0,1 mask
            pred_class = (pred_labels==cls_id).view(batch_size, -1).type(torch.FloatTensor) # 0,1 mask
            true_pix = torch.sum(class_mask, dim=1)
            pred_pix = torch.sum(pred_class, dim=1)

            for b in range(batch_size):
                if pred_pix[b] == 0 and true_pix[b] == 0:
                    batch_result.append(1.0)
                elif pred_pix[b] == 0 and true_pix[b] != 0:
                    batch_result.append(0.0)
                elif pred_pix[b] != 0 and true_pix[b] == 0:
                    batch_result.append(0.0)
                else:
                    batch_result.append((torch.sum(class_mask[b]*pred_class[b])/true_pix[b]).item())
                
            if size_average:
                acc["class_{}".format(cls_id)] = sum(batch_result)/batch_size
            else:
                acc["class_{}".format(cls_id)] = batch_result

        if size_average:
            acc["mean_all_pixel"] = torch.mean(torch.mean((pred_labels.type(torch.LongTensor)==gt_labels.type(torch.LongTensor)).view(batch_size, -1).type(torch.FloatTensor), dim=1), dim=0).item()
        else:
            result = []
            batch_result = torch.mean((pred_labels.type(torch.LongTensor)==gt_labels.type(torch.LongTensor)).view(batch_size, -1).type(torch.FloatTensor), dim=1)

            for b in range(batch_size):
                result.append(batch_result[b].item())

            acc["mean_all_pixel"] = result

        return acc

    else:
        batch_size = pred_labels.shape[0]
        w = pred_labels.shape[1]
        h = pred_labels.shape[2]
        
        # same thing.
        if size_average:
            return torch.mean(torch.mean((pred_labels.type(torch.LongTensor)==gt_labels).view(batch_size, -1).type(torch.FloatTensor), dim=1), dim=0).item()
        else:
            result = []
            batch_result = torch.mean((pred_labels.type(torch.LongTensor)==gt_labels).view(batch_size, -1).type(torch.FloatTensor), dim=1)
            for b in range(batch_size):
                result.append(batch_result[b].item())

            return result

def precision(pred_labels, gt_labels, class_num=2, size_average=True):
    result = {}
    batch_size = pred_labels.shape[0]

    for class_id in range(class_num):
        batch_result = []

        class_mask = (gt_labels==class_id).view(batch_size, -1).type(torch.LongTensor) # 0,1 mask
        pred_class = (pred_labels==class_id).view(batch_size, -1).type(torch.LongTensor) # 0,1 mask
        cls_gt = torch.sum(class_mask, dim=1)

        TP = torch.sum((pred_class.type(torch.LongTensor)*class_mask).view(batch_size, -1), dim=1).type(torch.FloatTensor)
        TPFP = torch.sum(pred_class.type(torch.LongTensor).view(batch_size, -1), dim=1).type(torch.FloatTensor)

        # to avoid error at all-zero mask
        for batch_index in range(batch_size):
            if TPFP[batch_index] == 0 and cls_gt[batch_index] == 0: 
                batch_result.append(1.0)
            elif TPFP[batch_index] == 0 and cls_gt[batch_index] != 0:
                batch_result.append(0.0)
            else:
                batch_result.append((TP[batch_index]/TPFP[batch_index]).item())

        if size_average:
            result["class_{}".format(class_id)] = sum(batch_result)/batch_size
        else:
            result["class_{}".format(class_id)] = batch_result

    return result

def dice_score(pred_labels, gt_labels):
    """
        binary class only
        return the accuracy of all pixels
        pred_labels and gt_labels should be batch x w x h
    """
    result = []

    w = pred_labels.size()[1]
    h = pred_labels.size()[2]
    pix = w*h
    batch_size = pred_labels.size()[0]

    TP = torch.sum((pred_labels.type(torch.LongTensor)*gt_labels).view(batch_size, pix), dim=1).type(torch.FloatTensor)
    FPFN = torch.sum((pred_labels.type(torch.LongTensor)!=gt_labels).view(batch_size, pix), dim=1).type(torch.FloatTensor)

    # to avoid error at all-zero mask
    for batch_index in range(batch_size):
        denominator = TP[batch_index]*2+FPFN[batch_index]
        if denominator == 0:
            result.append(1.0)
        else:
            result.append((TP[batch_index]*2)/denominator)

    return torch.FloatTensor(result)

def jaccard_index(pred_labels, gt_labels, class_num=2, size_average=True, only_class=None):
    """
        binary class only
        return the accuracy of all pixels
        pred_labels and gt_labels should be batch x w x h

        known as IoU
    """
    result = {}
    batch_size = pred_labels.shape[0]
    
    if only_class is not None:
        assert isinstance(only_class, int), "only_class should int"

        batch_result = []

        class_id = only_class

        class_mask = (gt_labels==cls_id).view(batch_size, pix).type(torch.FloatTensor) # 0,1 mask
        pred_class = (pred_labels==cls_id).view(batch_size, pix).type(torch.FloatTensor) # 0,1 mask
        pix = torch.sum(class_mask)
        
        TP = torch.sum((pred_class.type(torch.LongTensor)*class_mask).view(batch_size, -1), dim=1).type(torch.FloatTensor)
        P_1 = torch.sum(pred_class.type(torch.LongTensor).view(batch_size, -1), dim=1).type(torch.FloatTensor)
        G_1 = torch.sum(class_mask.type(torch.LongTensor).view(batch_size, -1), dim=1).type(torch.FloatTensor)

        # to avoid error at all-zero mask
        for batch_index in range(batch_size):
            denominator = P_1[batch_index] + G_1[batch_index] - TP[batch_index]
            if denominator == 0:
                batch_result.append(1.0)
            elif P_1 > 0 and G_1 == 0:
                batch_result.appedn(0.0)
            else:
                batch_result.append((TP[batch_index]/denominator).item())

        if size_average:
            result["class_{}".format(class_id)] = sum(batch_result)/batch_size
        else:
            result["class_{}".format(class_id)] = batch_result

    else:
        for class_id in range(class_num):
            batch_result = []

            class_mask = (gt_labels==class_id).view(batch_size, -1).type(torch.LongTensor) # 0,1 mask
            pred_class = (pred_labels==class_id).view(batch_size, -1).type(torch.LongTensor) # 0,1 mask

            TP = torch.sum((pred_class*class_mask).view(batch_size, -1), dim=1).type(torch.FloatTensor)
            P_1 = torch.sum(pred_class.view(batch_size, -1), dim=1).type(torch.FloatTensor)
            G_1 = torch.sum(class_mask.view(batch_size, -1), dim=1).type(torch.FloatTensor)

            # to avoid error at all-zero mask
            for batch_index in range(batch_size):
                denominator = P_1[batch_index] + G_1[batch_index] - TP[batch_index]
                if denominator == 0:
                    batch_result.append(1.0)
                elif P_1[batch_index] > 0 and G_1[batch_index] == 0:
                    batch_result.append(0.0)
                else:
                    batch_result.append((TP[batch_index]/denominator).item())

            if size_average:
                result["class_{}".format(class_id)] = sum(batch_result)/batch_size
            else:
                result["class_{}".format(class_id)] = batch_result

    return result

# from http://forums.fast.ai/t/understanding-the-dice-coefficient/5838
class SoftDiceLoss(nn.Module):
    """
        pred is size of batch x 2 channel x w x h
    """
    def __init__(self, weight=None, size_average=True, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def dice_coefficient(self, pred, target):
        batch_size = pred.shape[0]
        p = pred[:, 1].contiguous().view(batch_size, -1)
        t = target.view(batch_size, -1)
        
        # |p and t| => TP
        intersection = (p*t).sum(dim=1)
        #       2*TP                          /    |p| U |t| => 2*TP + FN + FP
        return (2.*intersection +self.smooth) / (p.sum(dim=1)+t.sum(dim=1)+self.smooth)

    def forward(self, pred, targets):
        batch_size = targets.shape[0]

        # dice score
        dice_loss = self.dice_coefficient(pred, targets)
        dice_loss = (1 - dice_loss).sum()/batch_size

        return dice_loss

# under here, not using 
################################################################################3
def seg_metric(pred_labels, gt_labels, class_num):
    """
        adding 1 to set all class incluging background to
        make correct class subtract to 0, and count the non-zero value
    
        input should be batch_size * height * width
    """    
    pix = pred_labels.shape[1]*pred_labels.shape[2]
    all_pix_acc = ((pred_labels-gt_labels) == 0).sum()/pix

    mean_cls_acc = 0

    for cls in range(class_num):
        # extract the indice of the labels(number) of cls
        mask = gt_labels == cls

        """
        # count cls-th class labels
        cls_pix_num = mask.sum()
        
        # count predicted class that belongs to cls-th class, which mean correct.
        pred_correct_cls = (pred_labels[mask] == cls).sum()
        
        class_predict_acc = pred_correct_cls/cls_pix_num
        """
        mean_cls_acc += (pred_labels[mask] == cls).sum()/mask.sum()
    
    return {"pix acc": all_pix_acc, "mean pix acc" : mean_cls_acc/class_num, "mean IoU":0}

def evaluate(label_trues, label_preds, n_class, eval_list=["IoU"]):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,}, cls_iu

if __name__ == '__main__':
    input_tensor = [
                    [[0,1,1],
                     [0,2,2],
                     [1,1,2]],
                    [[1,1,1],
                     [0,1,0],
                     [0,0,0]]
                    ]
    gt_tensor = [
                    [[1,1,1],
                     [0,2,2],
                     [0,0,2]],
                    [[2,2,2],
                     [2,2,2],
                     [2,2,2]]
                ]

    p = torch.LongTensor(input_tensor)
    g = torch.LongTensor(gt_tensor)
    results = [
                "pixel accuracy",
                pixel_accuracy(p, g, class_num=3, class_acc=False),
                pixel_accuracy(p, g, class_num=3, class_acc=False, size_average=False),
                pixel_accuracy(p, g, class_num=3, class_acc=True),
                pixel_accuracy(p, g, class_num=3, class_acc=True, size_average=False),
                "precision",
                precision(p, g, class_num=3, size_average=True),
                precision(p, g, class_num=3, size_average=False),
                "jaccard index",
                jaccard_index(p, g, class_num=3, size_average=True),
                jaccard_index(p, g, class_num=3, size_average=False)
              ]

    print(p)
    print(g)
    for result in results:
        print(result)
