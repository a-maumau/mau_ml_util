import yaml

def read_arguments_from_yaml(args, config_path):
    # config file must written in yaml style
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f)

        for config_arg in config["arguments"]:
            (key, val) = onfig_arg.items()[0]
            if key in args:
                args[key] = val
    except Exception as e:
        print(e)
        print("cannot read config file.")
