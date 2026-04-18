import inspect


class Singleton(type):
    _instances = {}
    _init = {}

    def __init__(cls, name, bases, dct):
        cls._init[cls] = dct.get("__init__", None)

    def __call__(cls, *args, **kwargs):
        init = cls._init[cls]
        if init is not None:
            args_list = list(args)
            for idx, arg in enumerate(args_list):
                args_list[idx] = str(arg)
            tmp_kwargs = {}
            for arg_key, arg_value in kwargs.items():
                tmp_kwargs[arg_key] = str(arg_value)
            key = (
                cls,
                frozenset(
                    inspect.getcallargs(init, None, *args_list, **tmp_kwargs).items()
                ),
            )
        else:
            key = cls

        if cls not in cls._instances:
            cls._instances[cls] = {}
        if key not in cls._instances[cls]:
            cls._instances[cls][key] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls][key]
