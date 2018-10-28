"""
    Pytorch is expected.
"""
import torch

CPU = torch.device('cpu')
NAN = float('nan')

class ClassificationMetric(object):
    """
        the matrix is holding the value in order
                ground truth
                0 1 2 ...
        pred  0
              1
              2
              .
              .
              .

            FP is X 
               |0 1 2
            ---+------
             0 |* X X
             1 |* * *
             2 |* * *

            FN is X
               |0 1 2
            ---+------
             0 |* * *
             1 |X * *
             2 |X * *
    """

    def __init__(self, class_num, map_device=CPU):
        self.class_num = class_num
        self.map_device = map_device

        self.confusion_matrix = torch.zeros(self.class_num, self.class_num).to(dtype=torch.long, device=self.map_device)

    def __call__(self, pred_labels, gt_labels):
        self.__add_to_matrix(pred_labels, gt_labels)

    # per batch
    def __add_to_matrix(self, pred_label, gt_label):
        for p_index in range(self.class_num):
            for gt_index in range(self.class_num):
                self.confusion_matrix[p_index, gt_index] += torch.sum((pred_label==p_index)*(gt_label==gt_index))

    def calc_acc(self):
        return float(torch.trace(self.confusion_matrix).cpu().item())/float(torch.sum(self.confusion_matrix).cpu().item())

    def calc_recall(self, ignore=[255]):
        """
            it is same to IoU
        """

        if isinstance(ignore, int):
            ignore = [ignore]
        recall = {}

        for class_id in range(self.class_num):
            if class_id in ignore:
                continue

            tpfn = torch.sum(self.confusion_matrix[:, class_id]).cpu().item()
            if tpfn == 0:
                recall["class_{}".format(class_id)] = NAN
            else:
                recall["class_{}".format(class_id)] = (float(self.confusion_matrix[class_id, class_id].cpu().item())/float(tpfn))

        return recall

    def calc_precision(self, ignore=[255]):
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
if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', action="store_true", default=False, help='')
    args = parser.parse_args()

    if args.gpu:
        map_device = torch.device('cuda')
    else:
        map_device = torch.device('cpu')

    """
        following tensors outputs should be like this

        batch0 p: class0=0.5, class1=0.5, class2=1.0
        batch1 p: class0=0.0, class1=0.0, class2=0.0
        mean   p: class0=0.25, class1=0.25, class2=0.5

        batch0 j: class0=0.25, class1=0.4, class2=1.0
        batch1 j: class0=0.0, class1=0.0, class2=0.0
        mean   j: class0=0.125, class1=0.2, class2=0.5
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

    print("prediction tensor")
    print(p)
    print("ground truth tensor")
    print(g)

    m = SegmentationMetric(3)
    m(p, g)

    print(m.confusion_matrix)
    print(m.calc_pix_acc())
    print(m.calc_mean_pix_acc())
    print(m.calc_mean_jaccard_index())
    print(m.calc_mean_precision())


    """
    results = [
                "pixel accuracy",
                pixel_accuracy(p, g, map_device=map_device),
                pixel_accuracy(p, g, size_average=False, map_device=map_device),
                "precision",
                precision(p, g, class_num=3, size_average=True, map_device=map_device),
                precision(p, g, class_num=3, size_average=False, map_device=map_device),
                "jaccard index",
                jaccard_index(p, g, class_num=3, size_average=True, map_device=map_device),
                jaccard_index(p, g, class_num=3, size_average=False, map_device=map_device)
              ]

    print("prediction tensor")
    print(p)
    print("ground truth tensor")
    print(g)
    for result in results:
        print(result)

    # speed check
    p = torch.randint(0, 10, (16, 512, 512)).to(device=map_device, dtype=torch.long)
    g = torch.randint(0, 10, (16, 512, 512)).to(device=map_device, dtype=torch.long)
    start = time.time()
    results = [
                pixel_accuracy(p, g, map_device=map_device),
                precision(p, g, class_num=3, size_average=True, map_device=map_device),
                jaccard_index(p, g, class_num=3, size_average=True, map_device=map_device)
              ]
    elapsed_time = time.time() - start
    print ("elapsed_time:{} sec".format(elapsed_time))
    """