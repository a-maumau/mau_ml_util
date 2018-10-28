from ..train_logger import TrainLogger
from ..metric import SegmentationMetric
from ..templates.trainer_template import Template_Trainer

import os
from datetime import datetime

import torch 

import numpy as np
from tqdm import tqdm
from PIL import Image

CPU = torch.device('cpu')
torch.backends.cudnn.benchmark = True

class SegmentationTrainer(Template_Trainer):
    def __init__(self, args, model, optimizer, lr_policy, train_loader, val_loader, map_device=CPU):
        """
            args: namespace (parser.parse_args())

            model: torch.nn.module (inherit mau_ml_util.templates.model_template)

            optimizer: torch.optim

            lr_policy: mau_ml_util.policy.learning_rate_policy

            train_loader: torch.utils.data.Dataset
                loader must return (image: torch.float, mask: torch.long)

            val_loader: torch.utils.data.Dataset
                loader must return (image: torch.float, mask: torch.long, original_image: torch.long)

            map_device: torch.device
                data will be mapped at map_decice
        """

        self.args = args
        if len(self.args.notify_mention) > 0:
            self.args.notify_mention += " "

        # for loggin the training
        val_head = ["val_num", "mean_pixel_accuracy"]
        for i in range(self.args.class_num):
            val_head.append("mean_precision_class_{}".format(i))
        for i in range(self.args.class_num):
            val_head.append("mean_IoU_class_{}".format(i))
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

        self.cmap = self._gen_cmap()

        self.lr_policy = lr_policy
        self.iter_wise = self.lr_policy.iteration_wise

        print("\nsaving at {}\n".format(self.save_dir))

    # PASCAL VOC color maps
    # borrowed from https://gist.github.com/wllhf/a4533e0adebe57e3ed06d4b50c8419ae
    def _gen_cmap(self, class_num=255):
        def bitget(byteval, idx):
            return ((byteval & (1 << idx)) != 0)

        cmap = np.zeros((class_num, 3), dtype='uint8')
        for i in range(class_num):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

        return cmap

    def convert_to_color_map(self, img_array, color_map=None, class_num=255):
        """
            img_array: numpy.ndarray
                shape must be (width, height)
        """

        if color_map is None:
            color_map = self._gen_cmap()

        new_img = np.empty(shape=(img_array.shape[0], img_array.shape[1], color_map.shape[1]), dtype='uint8')

        for c in range(class_num):
            index = np.where(img_array == c)[:2]
            new_img[index] = color_map[c]

        return new_img

    def save_sample_image(self, img, pred_mask, gt_mask, val_num, batch_num, desc="", desc_items=[]):
        if img is not None:
            batch_size = img.shape[0]

            self.tlog.setup_output("val_{}_batch_{}_sample".format(val_num, batch_num))
                    
            for n in range(batch_size):
                self.tlog.pack_output(Image.fromarray(np.uint8(img[n].detach().numpy())))

                pred_img = np.uint8(pred_mask[n].squeeze(0).cpu().detach().numpy())
                self.tlog.pack_output(Image.fromarray(pred_img), not_in_schema=True)
                self.tlog.pack_output(Image.fromarray(self.convert_to_color_map(pred_img, self.cmap)))

                gt_img = np.uint8(gt_mask[n].cpu().detach().numpy())
                self.tlog.pack_output(Image.fromarray(gt_img), not_in_schema=True)
                self.tlog.pack_output(Image.fromarray(self.convert_to_color_map(gt_img, self.cmap)))

                self.tlog.pack_output(None, desc, desc_items)
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
        pix_acc = 0.0
        precision_class = []
        jaccard_class = []
        
        metric = SegmentationMetric(self.args.class_num, map_device=self.map_device)
        _trainval_loader = self.to_tqdm(self.val_loader, desc="train val", quiet=self.args.quiet)

        for b, (image, mask, original_image) in enumerate(_trainval_loader):
            batch_size = image.shape[0]

            img = self.format_tensor(image, requires_grad=False, map_device=self.map_device)
            mask = self.format_tensor(mask, requires_grad=False, map_device=self.map_device)

            outputs = self.model.inference(img)

            metric(outputs, mask)
             
            # save only few batch for sample
            if b < self.args.validation_sample_batch_num:
                self.save_sample_image(original_image, outputs, mask, val_num, b, desc=" ")

        # flush the things that is packed in train logger
        self.save_sample_image(None, None, None, val_num, b, desc="validation sample", desc_items=["left: input", "center: pred cmap", "right: GT cmap"])

        # cals metrics
        pix_acc = metric.calc_pix_acc()
        precision = metric.calc_mean_precision()
        jaccard_index = metric.calc_mean_jaccard_index()

        for class_id in range(self.args.class_num):
            precision_class.append(precision["class_{}".format(class_id)])
            jaccard_class.append(jaccard_index["class_{}".format(class_id)])
        
        # clac. with out background
        log_msg_data = [val_num, pix_acc, np.mean(precision_class[1:]), np.mean(jaccard_class[1:])]
        # logging
        self.tlog.log("val", [val_num, pix_acc]+precision_class+jaccard_class)
        self.tlog.log_message("[#{: 6d}] mean pix acc.:{:.5f}, precision:{:.5f}, IoU:{:.5f}".format(*log_msg_data), "LOG", "validation")
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
            _train_loader = self.to_tqdm(self.train_loader, desc="", quiet=self.args.quiet):

            for img, mask in _train_loader:
                self.optimizer.zero_grad()

                images = self.format_tensor(img, map_device=self.map_device)
                masks = self.format_tensor(mask, map_device=self.map_device)

                output = self.model(images)
                batch_loss = self.model.loss(output, masks)

                total_loss += batch_loss.item()
                iter_total_loss += batch_loss.item()

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

        print("training data is saved at {}".format(self.save_dir))
        self.tlog.notify("{}{}: train finished.".format(self.args.notify_mention, self.args.save_name))
