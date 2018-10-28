"""
    parse and create arguments for policy
"""

def create_learning_rate_policy_arguments(optimizer, args):
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
