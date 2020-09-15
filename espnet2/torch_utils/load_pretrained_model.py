from pathlib import Path
from typing import Any
from typing import Union

import os
import logging
from collections import OrderedDict

import torch
import torch.nn
import torch.optim


def load_pretrained_model(
    pretrain_path: Union[str, Path],
    model: torch.nn.Module,
    pretrain_key: str = None,
    map_location: str = "cpu",
    ignore_not_existing_keys: bool = True,
):
    """Load a model state and set it to the model.

    Examples:
        >>> load_pretrained_model("somewhere/model.pth", model)
        >>> load_pretrained_model("somewhere/encoder.pth", model, "encoder")
    """
    if pretrain_key is None:
        obj = model
    else:

        def get_attr(obj: Any, key: str):
            """Get an nested attribute.

            >>> class A(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(10, 10)
            >>> a = A()
            >>> assert A.linear.weight is get_attr(A, 'linear.weight')

            """
            if key.strip() == "":
                return obj
            for k in key.split("."):
                obj = getattr(obj, k)
            return obj

        obj = get_attr(model, pretrain_key)

    state_dict = obj.state_dict()
    pretrained_dict = torch.load(pretrain_path, map_location=map_location)
    if ignore_not_existing_keys:
        # Ignores the parameters not existing in the train-model
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items() if k in state_dict
        }
    state_dict.update(pretrained_dict)
    obj.load_state_dict(state_dict)


def filter_modules(model_state_dict, modules):
    """Filter non-matched modules in module_state_dict.
    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer
    Return:
        new_mods (list): the update module list
    """
    new_mods = []
    incorrect_mods = []

    mods_model = list(model_state_dict.keys())
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]
        else:
            incorrect_mods += [mod]

    if incorrect_mods:
        logging.info(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.info("for information, the existing modules in model are: %s",
                     mods_model)

    return new_mods


def get_partial_state_dict(model_state_dict, modules):
    """Create state_dict with specified modules matching input model modules.
    Note that get_partial_lm_state_dict is used if a LM specified.
    Args:
        model_state_dict (OrderedDict): trained model state_dict
        modules (list): specified module list for transfer
    Return:
        new_state_dict (OrderedDict): the updated state_dict
    """
    new_state_dict = OrderedDict()

    for key, value in model_state_dict.items():
        if any(key.startswith(m) for m in modules):
            new_state_dict[key] = value

    return new_state_dict


def transfer_verification(model_state_dict, partial_state_dict, modules):
    """Verify tuples (key, shape) for input model modules match specified modules.
    Args:
        model_state_dict (OrderedDict): the initial model state_dict
        partial_state_dict (OrderedDict): the trained model state_dict
        modules (list): specified module list for transfer
    Return:
        (boolean): allow transfer
    """
    modules_model = []
    partial_modules = []

    for key_p, value_p in partial_state_dict.items():
        if any(key_p.startswith(m) for m in modules):
            partial_modules += [(key_p, value_p.shape)]

    for key_m, value_m in model_state_dict.items():
        if any(key_m.startswith(m) for m in modules):
            modules_model += [(key_m, value_m.shape)]

    len_match = len(modules_model) == len(partial_modules)

    module_match = sorted(modules_model, key=lambda x:
                          (x[0], x[1])) == sorted(partial_modules,
                                                  key=lambda x: (x[0], x[1]))

    return len_match and module_match


def load_pretrained_modules(
    model: torch.nn.Module,
    enc_model_path: Union[str, Path],
    dec_model_path: Union[str, Path],
    enc_modules: str = None,
    dec_modules: str = None,
    pretrain_key: str = None,
    map_location: str = "cpu",
):

    def print_new_keys(state_dict, modules, model_path):
        logging.info("loading %s from model: %s", modules, model_path)

        for k in state_dict.keys():
            logging.info("override %s" % k)

    logging.info(f"map location: {map_location}")
    state_dict = model.state_dict()
    for model_path, modules in [
        (enc_model_path, enc_modules),
        (dec_model_path, dec_modules),
    ]:
        if model_path and os.path.isfile(model_path):
            model_state_dict = torch.load(model_path, map_location=map_location)
            modules = filter_modules(model_state_dict, modules)
        partial_state_dict = get_partial_state_dict(model_state_dict, modules)
        if partial_state_dict:
            if transfer_verification(state_dict, partial_state_dict, modules):
                print_new_keys(partial_state_dict, modules, model_path)
                state_dict.update(partial_state_dict)
            else:
                logging.warning(
                    f"modules {modules} in model {model_path} "
                    f"don't match your training config",)
    model.load_state_dict(state_dict)
    return model