import yaml

def read_arguments_from_yaml(args, config_path):
    # config file must written in yaml style
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f)

        for config_arg in config["arguments"]:
            print(config_arg.items())
            (key, val) = config_arg.items()
            if key in args:
                args[key] = val
    except Exception as e:
        print(e)
        print("cannot read config file.")
