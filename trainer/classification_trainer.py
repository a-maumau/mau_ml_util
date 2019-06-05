from ..train_logger import TrainLogger
from ..metric import ClassificationMetric
from ..template.trainer_template import Template_Trainer
from ..utils.image_util import torch_tensor_to_image
from ..constant.constants import CPU

import os
from datetime import datetime

import torch 

import numpy as np
from tqdm import tqdm
from PIL import Image

torch.backends.cudnn.benchmark = True

class ClassificationTrainer(Template_Trainer):
    def __init__(self, args, model, optimizer, lr_policy, train_loader, val_loader, map_device=CPU):
        """
            args: namespace (parser.parse_args())

            model: torch.nn.module (inherit mau_ml_util.templates.model_template)

            optimizer: torch.optim

            lr_policy: mau_ml_util.policy.learning_rate_policy

            train_loader: torch.utils.data.Dataset
                loader must return (image: torch.float, label: tuple)

            val_loader: torch.utils.data.Dataset
                loader must return (image: torch.float, label: tuple, original_image: torch.long)

            map_device: torch.device
                data will be mapped at map_decice
        """

        self.args = args
        if len(self.args.notify_mention) > 0:
            self.args.notify_mention += " "

        # for loggin the training
        val_head = ["val_num", "accuracy", "precision", "recall"]
        for i in range(self.args.class_num):
            val_head.append("precision_class_{}".format(i))
        for i in range(self.args.class_num):
            val_head.append("recall_class_{}".format(i))
        self.tlog = self.get_train_logger({"train":["num", "batch_mean_total_loss"], "val":val_head},
                                          save_dir=self.args.save_dir, save_name=self.args.save_name, arguments=self.get_argparse_arguments(self.args),
                                          use_http_server=self.args.use_http_server, use_msg_server=self.args.use_msg_server, notificate=True,
                                          visualize_fetch_stride=self.args.viz_fetch_stride, http_port=self.args.http_server_port, msg_port=self.args.msg_server_port)        
        self.tlog.set_notificator(self.args.notify_type)
        self.tlog.notify("{}{}: initializing".format(self.args.notify_mention, self.args.save_name))

        # paths
        self.save_dir = self.tlog.log_save_path
        self.model_param_dir = self.tlog.mkdir("model_param")

        self.model = model
        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.map_device = map_device

        self.lr_policy = lr_policy
        self.iter_wise = self.lr_policy.iteration_wise

        print("\nsaving at {}\n".format(self.save_dir))

    def save_sample_image(self, img, pred_label, gt_label, val_num, batch_num, desc="", desc_items=[]):
        if img is not None:
            batch_size = img.shape[0]

            self.tlog.setup_output("val_{}_batch_{}_sample".format(val_num, batch_num))
                    
            for n in range(batch_size):
                pil_img = torch_tensor_to_image(img[n], coeff=255)
                self.tlog.pack_output(pil_img, not_in_schema=True,
                                      additional_name="input_gt_{}_pred_{}".format(int(pred_label[n].cpu().detach().item()), int(gt_label[n].cpu().detach().item())))

                self.tlog.pack_output(None, desc.format(int(gt_label[n].cpu().detach().item()), int(pred_label[n].cpu().detach().item())), desc_items)
        else:
            # flush the image that is packed in train logger
            self.tlog.pack_output(None, desc, desc_items)
            self.tlog.flush_output()

    # callback for each iteration or epoch
    def check_regulation(self, wise_type, decay_arg, num, total_loss, data_num, show_log_every=1):
        # decay
        self.lr_policy.decay_lr(**decay_arg)

        if num % show_log_every == 0:
            # logging
            self.tlog.log("train", [num, float(total_loss/data_num)])
            self.tlog.log_message("[{}: {: 6d}] batch mean loss:{:.5f}".format(wise_type, num, float(total_loss/data_num)), "LOG", "train")
            if not self.args.quiet:
                tqdm.write("[#{: 6d}] batch mean loss: {:.5f}".format(num, total_loss/data_num))

        # check train validation
        if num % self.args.trainval_every == 0:
            self.validate(num)

        # save model
        if num % self.args.save_every == 0:
            self.save_model(num, self.model, optimizer=self.optimizer, state={}, save_dir=self.model_param_dir, save_index_name=wise_type)
            self.tlog.log_message("[{}: {: 6d}] model saved.".format(wise_type, num), "LOG", "train")

    def validate(self, val_num):
        self.model.eval()

        # for logging
        acc = 0.0
        precision_class = []
        recall_class = []
        
        metric = ClassificationMetric(self.args.class_num, map_device=self.map_device)
        _trainval_loader = self.to_tqdm(self.val_loader, desc="train val", quiet=self.args.quiet)

        for b, (image, label) in enumerate(_trainval_loader):
            batch_size = image.shape[0]

            images = self.format_tensor(image, requires_grad=False, map_device=self.map_device)
            labels = self.format_tensor(torch.LongTensor(label), requires_grad=False, map_device=self.map_device)

            outputs = self.model.inference(images)

            metric(outputs, labels)
             
            # save only few batch for sample
            if b < self.args.validation_sample_batch_num:
                self.save_sample_image(images, outputs, labels, val_num, b, desc="pred: {}, gt: {}")

        # flush the things that is packed in train logger
        self.save_sample_image(None, None, None, val_num, b, desc="validation sample", desc_items=["left: original input", "right: transformed"])

        # cals metrics
        acc = metric.calc_acc()
        precision = metric.calc_precision()
        recall = metric.calc_recall()

        for class_id in range(self.args.class_num):
            precision_class.append(precision["class_{}".format(class_id)])
            recall_class.append(recall["class_{}".format(class_id)])
        
        log_msg_data = [val_num, acc, np.mean(precision_class), np.mean(recall_class)]
        # logging
        self.tlog.log("val", log_msg_data+precision_class+recall_class)
        self.tlog.log_message("[#{: 6d}] mean pix acc.:{:.5f}, precision:{:.5f}, recall:{:.5f}".format(*log_msg_data), "LOG", "validation")
        if not self.args.quiet:
           tqdm.write("[#{: 6d}] mean pix acc.:{:.5f}, precision:{:.5f}, IoU:{:.5f}".format(*log_msg_data))

        self.model.train()

    def train(self):
        self.tlog.notify("{}{}: start training".format(self.args.notify_mention, self.args.save_name))

        data_num = len(self.train_loader)
        epoch_num = self.calc_iter_to_epoch(data_num, self.args.max_iter) if self.iter_wise else self.args.epochs
        epochs = self.to_tqdm(range(1, epoch_num+1), desc="[train::{}]".format(self.args.save_name), quiet=self.args.quiet)

        epoch = 0
        curr_iter = 0
        total_loss = 0.0

        # for epoch wise and iter wise
        # because of existing option of forcing iter/epoch wise, we need pass them both
        decay_arg = {"curr_iter":curr_iter, "curr_epoch":epoch}

        for epoch in epochs:
            _train_loader = self.to_tqdm(self.train_loader, desc="", quiet=self.args.quiet)

            for img, label in _train_loader:
                self.optimizer.zero_grad()

                images = self.format_tensor(img, map_device=self.map_device)
                labels = self.format_tensor(torch.LongTensor(label), map_device=self.map_device)

                output = self.model(images)
                batch_loss = self.model.loss(output, labels)

                total_loss += batch_loss.item()

                batch_loss.backward()
                self.optimizer.step()

                curr_iter += 1

                # for iter wise callback to check: to decay, show log, validate, save model
                if self.iter_wise:
                    self.check_regulation("iter", decay_arg, curr_iter, total_loss, self.args.show_log_every_iter, show_log_every=self.args.show_log_every_iter)

                    if curr_iter % self.args.show_log_every_iter == 0:
                        total_loss = 0.0
                        data_num = 0

                    if curr_iter == self.args.max_iter:
                        break

                # for printing batch result
                if not self.args.quiet:
                    lr = self.optimizer.param_groups[0]["lr"]
                    _train_loader.set_description("[#{: 6d}]::[lr: {:.4f}]: train[{}] loss: {:.5f}".format(epoch, lr, self.args.save_name, batch_loss))

            # for epoch wise callback to check: to decay, show log, validate, save model
            if not self.iter_wise:
                self.check_regulation("epoch", decay_arg, epoch, total_loss, data_num, show_log_every=1)

                total_loss = 0.0

        last_val = curr_iter if self.iter_wise else epoch
        index_namae = "iter" if self.iter_wise else "epoch"
        # save trained model
        self.save_model(last_val, self.model, optimizer=self.optimizer, state={},
                        save_dir=self.model_param_dir, save_name='model_param_fin_{}.pth'.format(datetime.now().strftime("%Y%m%d_%H-%M-%S")),
                        save_index_name=index_namae)

        print("\ntraining data is saved at {}\n".format(self.save_dir))
        self.tlog.notify("{}{}: train finished.".format(self.args.notify_mention, self.args.save_name))
