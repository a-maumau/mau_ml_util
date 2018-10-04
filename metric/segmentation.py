"""
    Pytorch is expected.
"""

import torch

CPU = torch.device('cpu')

class SegmentationMetric(object):
    def __init__(self, class_num, map_device=CPU):
        self.class_num = class_num
        self.map_device = map_device

        self.class_matrix = torch.zeros(self.class_num, self.class_num).to(device=map_device, dtype=torch.long)

    def __call__(self, pred_labels, gt_labels):
        batch_size = pred_labels.shape[0]

        for batch in range(batch_size):
            self.__add_to_matrix(pred_labels[batch], gt_labels[batch])

    # per batch
    def __add_to_matrix(self, pred_label, gt_label):
        for gt_class in range(self.class_num):
            tensor_2_class = torch.eq(gt_label, gt_class).to(dtype=torch.long)

            #print("top for")
            #print(tensor_2_class)

            for pred_class in range(self.class_num):
                tensor_1_class = torch.eq(pred_label, pred_class).to(dtype=torch.long)
                #print("intra for")
                #print(tensor_1_class)
                tensor_1_class = torch.mul(tensor_2_class, tensor_1_class)
                #print(tensor_1_class)
                count = torch.sum(tensor_1_class)
                #print(count)
                self.class_matrix[gt_class, pred_class] +=count

    def l(self, pred_labels, gt_labels):
        batch_size = pred_labels.shape[0]
        class_matrix = torch.zeros(self.class_num, self.class_num).to(device=map_device, dtype=torch.long)

        def hist_per_batch(tensor_1, tensor_2, ignore_label=255, classes=3):
            hist_tensor = torch.zeros(classes,classes).to(device=map_device, dtype=torch.long)
            for class_2_int in range(classes):
                tensor_2_class = torch.eq(tensor_2,class_2_int).long()
                for class_1_int in range(classes):
                    tensor_1_class = torch.eq(tensor_1,class_1_int).long()
                    tensor_1_class = torch.mul(tensor_2_class,tensor_1_class)
                    count = torch.sum(tensor_1_class)
                    hist_tensor[class_2_int,class_1_int] +=count

            return hist_tensor

        for batch in range(batch_size):
            a = hist_per_batch(pred_labels[batch], gt_labels[batch])
            class_matrix = torch.add(class_matrix, a)

        return class_matrix

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
    input_tensor = [
                    [[0,1,1],
                     [0,2,2],
                     [1,1,2]],
                    ]
    gt_tensor = [
                    [[1,1,1],
                     [0,2,2],
                     [0,0,2]],
                ]

    p = torch.LongTensor(input_tensor).to(device=map_device)
    g = torch.LongTensor(gt_tensor).to(device=map_device)

    print("prediction tensor")
    print(p)
    print("ground truth tensor")
    print(g)

    m = SegmentationMetric(3)
    m(p, g)

    print(m.class_matrix)
    print(m.l(p, g))

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