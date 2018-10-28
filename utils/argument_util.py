import yaml

def read_arguments_from_yaml(args, config_path):
    # config file must written in yaml style
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f)

        for config_arg in config["arguments"]:
            if config_arg.key() in args:
                args[config_arg.key()] = config_arg.value()
    except:
        print("cannot read config file.")
