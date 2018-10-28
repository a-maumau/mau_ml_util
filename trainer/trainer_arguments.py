"""
    argments which required in trainers (also other dependency files).
"""

def trainer_arguments(parser):
    # saveing setting
    parser.add_argument('--save_name', type=str, default="train", help='dir of saving log and model parameters and so on.')
    parser.add_argument('--save_dir', type=str, default="log", help='dir of saving log and model parameters and so on.')
    parser.add_argument('--save_every', type=int, default=10, help='count of saving model.')
    
    # logging count
    parser.add_argument('--trainval_every', type=int, default=5, help="count of evaluate training validation.")
    parser.add_argument('--show_log_every_iter', type=int, default=1000, help="count of showing the log for iter style.")
    parser.add_argument('--validation_sample_batch_num', type=int, default=1, help='num of saving sample in validation. value denotes batch num.')

    # training setting
    parser.add_argument('--epochs', type=int, default=100, help="train epochs.")
    parser.add_argument('--max_iter', type=int, default=200000, help="train iter max num.")
    parser.add_argument('--decay_every', type=int, default=30, help="count of decaying learning rate.")
    parser.add_argument('--batch_size', type=int, default=4, help="mini batch size.")
    parser.add_argument('--num_workers', type=int, default=4, help="worker number of data loader.")
    parser.add_argument('--class_num', type=int, default=21, help="class number for predicting,")

    parser.add_argument('--learning_rate', type=float, default=0.001, help="initial value of learning rate.")
    parser.add_argument('--min_learning_rate', type=float, default=0.001, help="initial value of learning rate.")
    parser.add_argument('--lr_decay_power', type=float, default=0.9, help="count of decaying learning rate.")
    parser.add_argument('--decay_value', type=float, default=0.1, help="decay learning rate with count of args:decay_every in this factor.")
    parser.add_argument('--lr_hp_k', type=float, default=1.0, help="hyper parameter in lernaing rate policy.")

    # gpu number
    parser.add_argument('--gpu_device_num', type=int, default=0, help="device number of gpu.")

    # setting of visualization
    parser.add_argument('--viz_fetch_stride', type=int, default=1, help="")
    parser.add_argument('--http_server_port', type=int, default=8080, help="")
    parser.add_argument('--msg_server_port', type=int, default=8081, help="")

    # notification
    parser.add_argument('--notify_mention', type=str, default="", help='if you want to set a mention. ex twitter, @user_name')
    parser.add_argument('--notify_type', type=str, nargs='*', default=["slack"], help="you can pick up multiple type from [slack mail twitter].")

    # flag option
    parser.add_argument('-nogpu', action="store_true", default=False, help="don't use gpu")
    parser.add_argument('-show_parameters', action="store_true", default=False, help='show model parameters')
    parser.add_argument('-quiet', action="store_true", default=False, help='only showing the log of loss and validation')
    parser.add_argument('-use_http_server', action="store_true", default=False, help='')
    parser.add_argument('-use_msg_server', action="store_true", default=False, help='')

    parser.add_argument('-force_lr_policy_iter_wise', action="store_true", default=False, help='')
    parser.add_argument('-force_lr_policy_epoch_wise', action="store_true", default=False, help='')

def change_default_trainer_arguments(args, config_path):
    # config file must written in yaml style
    pass
