import abc
import traceback

import torch
import torch.nn as nn

from tqdm import tqdm

from train_logger import TrainLogger

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

    def setup_train_logger(self, namespaces, save_dir="./", save_name="log", arguments=[],
                           use_http_server=True, use_msg_server=True, notificate=False,
                           visualize_fetch_stride=1, http_port=8080, msg_port=8081):
        # saving directory can get with save_dir = tlog.log_save_path
        tlog = TrainLogger(log_dir=save_dir, log_name=save_name, namespaces=namespaces,
                           arguments=arguments, notificate=notificate, suppress_err=False, visualize_fetch_stride=visualize_fetch_stride)
        tlog.start_http_server(bind_port=args.http_port)
        tlog.start_msg_server(bind_port=args.msg_port)

        return tlog

    def get_argparse_arguments(self, args):
        return args._get_kwargs()

    def format_tensor(self, x, requires_grad=True, nogpu=False, gpu_device_num=0):
        if torch.cuda.is_available() and not nogpu:
            if not requires_grad:
                x = x.cuda(gpu_device_num).detach()
            else:
                x = x.cuda(gpu_device_num)
        else:
            if not requires_grad:
                x = x.detach()

        return x

    def map_on_gpu(self, model, gpu_device_num=0):
        if torch.cuda.is_available():
            # for cpu, it is 'cpu', but default mapping is cpu.
            # so if you want use on cpu, just don't call this
            map_device = torch.device('cuda:{}'.format(gpu_device_num))
            model = model.to(map_device)

    def decay_learning_rate(self, optimizer, decay_value):
        for param_group in optimizer.param_groups:
                param_group['lr'] *= decay_value

    def to_tqdm(self, loader. desc=""):
        return tqdm(loader, desc=desc, ncols=self.tqdm_ncols)

class Template_Model(nn.Module):
    __metaclass__ = abc.ABCMeta

    def initialize_weights(self, *models):
        for model in models:
            for module in model.modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.init.kaiming_normal(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.ConvTranspose2d):
                    module.weight.data.normal_(0, 0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()
                    module.bias.data.zero_()

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def loss(self, inputs, targets):
        raise NotImplementedError()

    def inference(self, x):
        pass

    # for loading trained parameter of this model.
    def load_trained_param(self, parameter_path, print_debug=False):
        if parameter_path is not None:
            try:
                print("loading pretrained parameter... ", end="")
                
                chkp = torch.load(os.path.abspath(parameter_path), map_location=lambda storage, location: storage)

                if print_debug:
                    print(chkp.keys())

                self.load_state_dict(chkp["state_dict"])
                
                print("done.")

            except Exception as e:
                print("\n"+e+"\n")
                traceback.print_exc()
                print("cannot load pretrained data.")

    def save(self, add_state={}, file_name="model_param.pth"):
        #assert type(add_state) is dict, "arg1:add_state must be dict"
        
        if "state_dict" in add_state:
            print("the value of key:'state_dict' will be over write with model's state_dict parameters")

        _state = add_state
        _state["state_dict"] = self.state_dict()
        
        try:
            torch.save(_state, file_name)
        except:
            torch.save(self.state_dict(), "./model_param.pth.tmp")
            print("save_error.\nsaved at ./model_param.pth.tmp only model params.")
