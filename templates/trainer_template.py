from ..train_logger import TrainLogger
from ..utils.path_util import path_join

import abc

import torch
from tqdm import tqdm

CPU = torch.device('cpu')

class Template_Trainer:
    __metaclass__ = abc.ABCMeta

    tqdm_ncols = 100

    @abc.abstractmethod
    def validate(self):
        """
            in here, model should set at eval mode
            model.eval()

            after evaliation, set train mode
            model.train()
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError()

    def get_train_logger(self, namespaces, save_dir="./", save_name="log", arguments=[],
                           use_http_server=False, use_msg_server=False, notificate=False,
                           visualize_fetch_stride=1, http_port=8080, msg_port=8081):
        # saving directory can get with save_dir = tlog.log_save_path
        tlog = TrainLogger(log_dir=save_dir, log_name=save_name, namespaces=namespaces,
                           arguments=arguments, notificate=notificate, suppress_err=True, visualize_fetch_stride=visualize_fetch_stride)
        if use_http_server:
            tlog.start_http_server(bind_port=http_port)

        if use_msg_server:
            tlog.start_msg_server(bind_port=msg_port)

        return tlog

    def get_argparse_arguments(self, args):
        return args._get_kwargs()

    def format_tensor(self, x, requires_grad=True, map_device=CPU):            
        if requires_grad:
            x = x.to(map_device)
        else:
            x = x.to(map_device).detach()

        return x

    def map_on_gpu(self, model, gpu_device_num=0):
        if torch.cuda.is_available():
            # for cpu, it is 'cpu', but default mapping is cpu.
            # so if you want use on cpu, just don't call this
            map_device = torch.device('cuda:{}'.format(gpu_device_num))
            model = model.to(map_device)

    def calc_iter_to_epoch(self, epoch_batch_num, max_iter):
        e = max_iter//epoch_batch_num

        if max_iter % epoch_batch_num != 0:
            e += 1

        return e

    def decay_learning_rate(self, optimizer, decay_value):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_value

    def to_tqdm(self, loader, desc="", quiet=False):
        if quiet:
            return loader

        return tqdm(loader, desc=desc, ncols=self.tqdm_ncols)

    def save_model(self, save_index, model, optimizer=None, state={}, save_dir="./", save_name="model_param_e{}.pth", save_index_name="num"):
        if optimizer is not None:
            state['optimizer_state_dict'] = optimizer.state_dict()

        state[save_index_name] = save_index

        model.save(add_state=state, file_name=path_join(save_dir, save_name.format(save_index)))
