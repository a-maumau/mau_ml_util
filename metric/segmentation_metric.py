"""
    Pytorch is expected.
"""
from ..constant.constants import CPU, NAN

import torch

class SegmentationMetric(object):
    """
        the matrix is holding the value in order
                ground truth
                0 1 2 3 4 ...
        pred  0
              1
              2
              3
              4
              .
              .
              .

        so, in 3 class case with following tensor
        pred_tensor = [
                    [[0,1,1],
                     [0,2,2],
                     [1,1,2]],
                    ]
        gt_tensor = [
                    [[1,1,1],
                     [0,2,2],
                     [0,0,2]],
                    ]

        the matrix is goiing to be
               |ground truth
               |0 1 2
        -------+-----
        pred 0 |1 1 0
             1 |2 2 0
             2 |0 0 3

        and, to calc iou, for example,
        class 0:
            TP is 
               |0 1 2
            ---+-----
             0 |1 * *
             1 |* * *
             2 |* * *

            FP is 
               |0 1 2
            ---+-----
             0 |* 1 0
             1 |* * *
             2 |* * *

            FN is
               |0 1 2
            ---+-----
             0 |* * *
             1 |2 * *
             2 |0 * *

        so IoU of class 0 is TP/(TP+FP+FN) = 1/(1+1+2+) = 0.25

        and so on.
    """

    def __init__(self, class_num, map_device=CPU):
        self.class_num = class_num
        self.map_device = map_device

        self.confusion_matrix = torch.zeros(self.class_num, self.class_num).to(device=map_device, dtype=torch.long)

    def __call__(self, pred_labels, gt_labels):
        self.__add_to_matrix(pred_labels, gt_labels)

    # per batch
    def __add_to_matrix(self, pred_label, gt_label):
        for p_index in range(self.class_num):
            for gt_index in range(self.class_num):
                self.confusion_matrix[p_index, gt_index] += torch.sum((pred_label==p_index)*(gt_label==gt_index))
        """
        for gt_class_id in range(self.class_num):
            gt_class = torch.eq(gt_label, gt_class_id).to(dtype=torch.long)

            for pred_class_id in range(self.class_num):
                pred_class = torch.eq(pred_label, pred_class_id).to(dtype=torch.long)
                pred_class = torch.mul(gt_class, pred_class)
                count = torch.sum(pred_class)
                self.confusion_matrix[pred_class_id, gt_class_id] += count
        """

    def calc_pix_acc(self):
        return float(torch.trace(self.confusion_matrix).cpu().item())/float(torch.sum(self.confusion_matrix).cpu().item())

    def calc_mean_pix_acc(self, ignore=[255]):
        if isinstance(ignore, int):
            ignore = [ignore]
        mean_pix_acc = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            all_class_id_pix = torch.sum(self.confusion_matrix[:, class_id]).cpu().item()
            if all_class_id_pix == 0:
                mean_pix_acc["class_{}".format(class_id)] = NAN
            else:
                mean_pix_acc["class_{}".format(class_id)] = (float(self.confusion_matrix[class_id, class_id].cpu().item())/float(all_class_id_pix))/self.class_num

        return mean_pix_acc

    def calc_mean_jaccard_index(self, ignore=[255]):
        """
            it is same to IoU
        """

        if isinstance(ignore, int):
            ignore = [ignore]
        iou = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            tpfpfn = (torch.sum(self.confusion_matrix[class_id, :])+torch.sum(self.confusion_matrix[:, class_id])-self.confusion_matrix[class_id, class_id]).cpu().item()
            if tpfpfn == 0:
                iou["class_{}".format(class_id)] = NAN
            else:
                iou["class_{}".format(class_id)] = (float(self.confusion_matrix[class_id, class_id].cpu().item())/float(tpfpfn))

        return iou

    def calc_mean_precision(self, ignore=[255]):
        if isinstance(ignore, int):
            ignore = [ignore]
        precision = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            tpfp = torch.sum(self.confusion_matrix[class_id, :]).cpu().item()
            if tpfp == 0:
                precision["class_{}".format(class_id)] = NAN
            else:
                precision["class_{}".format(class_id)] = (float(self.confusion_matrix[class_id, class_id].cpu().item())/float(tpfp))

        return precision

# test
def __test_module(use_gpu=False):
    from ..test.test_util import print_test_starting, print_test_log, print_test_passed, print_test_failed, return_test_passed, return_test_failed

    print_test_starting(__file__)

    if use_gpu:
        map_device = torch.device('cuda')
    else:
        map_device = torch.device('cpu')

    """
        following tensors outputs should be like this
        pixel acc: 0.333...
                class        0         1         2
                    p: 0.1428571...   0.25      1.0
                    j: 0.111...       0.222...  0.25
         mean pix acc: 0.111...       0.222...  0.0833...
    """
    pred_tensor = [
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

    p = torch.LongTensor(pred_tensor).to(device=map_device)
    g = torch.LongTensor(gt_tensor).to(device=map_device)

    print_test_log("input tensor")
    print_test_log(p)
    print_test_log("")
    print_test_log("ground truth tensor")
    print_test_log(g)

    m = SegmentationMetric(3)
    m(p, g)

    metric_results = [m.calc_pix_acc(),
                      m.calc_mean_pix_acc(),
                      m.calc_mean_jaccard_index(),
                      m.calc_mean_precision()]

    # stupid test...
    print_test_log("confusion matrix:")
    print_test_log(m.confusion_matrix)
    
    try:
        if metric_results[0] != 0.3333333333333333:
            print_test_log("expect: {}, but {}".format(v, metric_results[0][k]))
            raise Exception("VALUE IS NOT CORRECT")

        correct_score = {'class_0':0.1111111111111111, 'class_1':0.2222222222222222, 'class_2':0.08333333333333333}
        for k, v in correct_score.items():
            if metric_results[1][k] != v:
                print_test_log("expect: {}, but {}".format(v, metric_results[1][k]))
                raise Exception("VALUE IS NOT CORRECT")

        correct_score = {'class_0': 0.1111111111111111, 'class_1': 0.2222222222222222, 'class_2': 0.25}
        for k, v in correct_score.items():
            if metric_results[2][k] != v:
                print_test_log("expect: {}, but {}".format(v, metric_results[2][k]))
                raise Exception("VALUE IS NOT CORRECT")

        correct_score = {'class_0': 0.14285714285714285, 'class_1': 0.25, 'class_2': 1.0}
        for k, v in correct_score.items():
            if metric_results[3][k] != v:
                print_test_log("expect: {}, but {}".format(v, metric_results[3][k]))
                raise Exception("VALUE IS NOT CORRECT")
    except:
        print_test_failed()
        return return_test_failed()
    
    print_test_passed()
    return return_test_passed()
