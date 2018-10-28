from ..train_logger import TrainLogger

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
        if not requires_grad:
            x = x.to(map_device).detach()
        else:
            x = x.to(map_device)

        return x

    def gen_policy_args(self, optimizer, args):
        policy_args = {"optimizer":optimizer}

        policy_args["initial_learning_rate"] = args.learning_rate
        policy_args["max_iter"] = args.max_iter
        policy_args["lr_decay_power"] = args.lr_decay_power
        policy_args["decay_epoch"] = args.decay_every
        policy_args["decay_val"] = args.decay_value
        policy_args["max_learning_rate"] = args.learning_rate
        policy_args["min_learning_rate"] = args.min_learning_rate
        policy_args["k"] = args.lr_hp_k

        if args.force_lr_policy_iter_wise and args.force_lr_policy_epoch_wise:
            pass
        elif args.force_lr_policy_iter_wise:
            policy_args["iteration_wise"] = True
        elif args.force_lr_policy_epoch_wise:
            policy_args["iteration_wise"] = False
        else:
            pass

        return policy_args

    def map_on_gpu(self, model, gpu_device_num=0):
        if torch.cuda.is_available():
            # for cpu, it is 'cpu', but default mapping is cpu.
            # so if you want use on cpu, just don't call this
            map_device = torch.device('cuda:{}'.format(gpu_device_num))
            model = model.to(map_device)

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

        model.save(add_state=state, file_name=os.path.join(save_dir, save_name.format(save_index)))
