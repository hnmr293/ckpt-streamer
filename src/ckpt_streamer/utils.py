import torch


def get_module_and_param(module: torch.nn.Module, key: str) -> tuple[torch.nn.Module, str, torch.Tensor]:
    if not isinstance(key, str):
        raise TypeError(f"key must be str, but got {key.__class__}")

    keys = key.rsplit(".", 1)

    if len(keys) != 2:
        raise ValueError(f"key must be in the form of 'module_name.param_name', but got {key!r}")

    mod_key, param_key = keys

    submod = module.get_submodule(mod_key)
    param = getattr(submod, param_key)

    if not isinstance(param, torch.Tensor):
        # TypeError が正しい気がするけど torch.get_submodule に合わせて AttributeError にしておく
        raise AttributeError(f"param must be torch.Tensor, but got {param.__class__}")

    return submod, param_key, param
