import lucent
import timm
from lucent.optvis import render, param, transform, objectives
from torchvision import transforms
from decorator import decorator

import torch


class Objective:
    def __init__(self, objective_func, name="", description=""):
        self.objective_func = objective_func
        self.name = name
        self.description = description

    def __call__(self, model):
        return self.objective_func(model)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other + self(model)
            name = self.name
            description = self.description
        else:
            objective_func = lambda model: self(model) + other(model)
            name = ", ".join([self.name, other.name])
            description = (
                "Sum(" + " +\n".join([self.description, other.description]) + ")"
            )
        return Objective(objective_func, name=name, description=description)

    @staticmethod
    def sum(objs):
        objective_func = lambda T: sum([obj(T) for obj in objs])
        descriptions = [obj.description for obj in objs]
        description = "Sum(" + " +\n".join(descriptions) + ")"
        names = [obj.name for obj in objs]
        name = ", ".join(names)
        return Objective(objective_func, name=name, description=description)

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            objective_func = lambda model: other * self(model)
            return Objective(
                objective_func, name=self.name, description=self.description
            )
        else:
            # Note: In original Lucid library, objectives can be multiplied with non-numbers
            # Removing for now until we find a good use case
            raise TypeError(
                "Can only multiply by int or float. Received type " + str(type(other))
            )

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return self.__mul__(1 / other)
        else:
            raise TypeError(
                "Can only divide by int or float. Received type " + str(type(other))
            )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __radd__(self, other):
        return self.__add__(other)


def _make_arg_str(arg):
    arg = str(arg)
    too_big = len(arg) > 15 or "\n" in arg
    return "..." if too_big else arg


def wrap_objective():
    @decorator
    def inner(func, *args, **kwds):
        objective_func = func(*args, **kwds)
        objective_name = func.__name__
        args_str = " [" + ", ".join([_make_arg_str(arg) for arg in args]) + "]"
        description = objective_name.title() + args_str
        return Objective(objective_func, objective_name, description)

    return inner


@wrap_objective()
def neuron(layer, n_channel, x=None, y=None, batch=None):
    @handle_batch(batch)
    def inner(model):
        layer_t = model(layer)
        layer_t = _extract_act_pos(layer_t, x)
        # return -layer_t[:, n_channel].mean()

        # print(f"Line 87, layer: {layer}")
        # print(f"Line 90, layer_t.shape[1]: {layer_t.shape[1]}")
        return -layer_t[:, n_channel].sum()

    return inner


def handle_batch(batch=None):
    return lambda f: lambda model: f(_T_handle_batch(model, batch=batch))


def _T_handle_batch(T, batch=None):
    def T2(name):
        t = T(name)
        if isinstance(batch, int):
            return t[batch : batch + 1]
        else:
            return t

    return T2


def _extract_act_pos(acts, x=None):
    shape = acts.shape
    x = shape[2] // 2 if x is None else x
    return acts[:, :, x : x + 1]


vit_transforms = [
    transform.pad(16),
    transform.jitter(8),
    transform.random_scale([n / 100.0 for n in range(80, 100)]),
    transform.random_rotate(
        list(range(-10, 10)) + list(range(-5, 5)) + 10 * list(range(-2, 2))
    ),
    transform.jitter(2),
    transforms.Resize((224, 224)),
]


all_layer_names = [
    "patch_embed",
    "patch_embed_proj",
    "patch_embed_norm",
    "pos_drop",
    "patch_drop",
    "norm_pre",
    "blocks",
    "blocks_0",
    "blocks_0_norm1",
    "blocks_0_attn",
    "blocks_0_attn_qkv",
    "blocks_0_attn_q_norm",
    "blocks_0_attn_k_norm",
    "blocks_0_attn_attn_drop",
    "blocks_0_attn_proj",
    "blocks_0_attn_proj_drop",
    "blocks_0_ls1",
    "blocks_0_drop_path1",
    "blocks_0_norm2",
    "blocks_0_mlp",
    "blocks_0_mlp_fc1",
    "blocks_0_mlp_act",
    "blocks_0_mlp_drop1",
    "blocks_0_mlp_norm",
    "blocks_0_mlp_fc2",
    "blocks_0_mlp_drop2",
    "blocks_0_ls2",
    "blocks_0_drop_path2",
    "blocks_1",
    "blocks_1_norm1",
    "blocks_1_attn",
    "blocks_1_attn_qkv",
    "blocks_1_attn_q_norm",
    "blocks_1_attn_k_norm",
    "blocks_1_attn_attn_drop",
    "blocks_1_attn_proj",
    "blocks_1_attn_proj_drop",
    "blocks_1_ls1",
    "blocks_1_drop_path1",
    "blocks_1_norm2",
    "blocks_1_mlp",
    "blocks_1_mlp_fc1",
    "blocks_1_mlp_act",
    "blocks_1_mlp_drop1",
    "blocks_1_mlp_norm",
    "blocks_1_mlp_fc2",
    "blocks_1_mlp_drop2",
    "blocks_1_ls2",
    "blocks_1_drop_path2",
    "blocks_2",
    "blocks_2_norm1",
    "blocks_2_attn",
    "blocks_2_attn_qkv",
    "blocks_2_attn_q_norm",
    "blocks_2_attn_k_norm",
    "blocks_2_attn_attn_drop",
    "blocks_2_attn_proj",
    "blocks_2_attn_proj_drop",
    "blocks_2_ls1",
    "blocks_2_drop_path1",
    "blocks_2_norm2",
    "blocks_2_mlp",
    "blocks_2_mlp_fc1",
    "blocks_2_mlp_act",
    "blocks_2_mlp_drop1",
    "blocks_2_mlp_norm",
    "blocks_2_mlp_fc2",
    "blocks_2_mlp_drop2",
    "blocks_2_ls2",
    "blocks_2_drop_path2",
    "blocks_3",
    "blocks_3_norm1",
    "blocks_3_attn",
    "blocks_3_attn_qkv",
    "blocks_3_attn_q_norm",
    "blocks_3_attn_k_norm",
    "blocks_3_attn_attn_drop",
    "blocks_3_attn_proj",
    "blocks_3_attn_proj_drop",
    "blocks_3_ls1",
    "blocks_3_drop_path1",
    "blocks_3_norm2",
    "blocks_3_mlp",
    "blocks_3_mlp_fc1",
    "blocks_3_mlp_act",
    "blocks_3_mlp_drop1",
    "blocks_3_mlp_norm",
    "blocks_3_mlp_fc2",
    "blocks_3_mlp_drop2",
    "blocks_3_ls2",
    "blocks_3_drop_path2",
    "blocks_4",
    "blocks_4_norm1",
    "blocks_4_attn",
    "blocks_4_attn_qkv",
    "blocks_4_attn_q_norm",
    "blocks_4_attn_k_norm",
    "blocks_4_attn_attn_drop",
    "blocks_4_attn_proj",
    "blocks_4_attn_proj_drop",
    "blocks_4_ls1",
    "blocks_4_drop_path1",
    "blocks_4_norm2",
    "blocks_4_mlp",
    "blocks_4_mlp_fc1",
    "blocks_4_mlp_act",
    "blocks_4_mlp_drop1",
    "blocks_4_mlp_norm",
    "blocks_4_mlp_fc2",
    "blocks_4_mlp_drop2",
    "blocks_4_ls2",
    "blocks_4_drop_path2",
    "blocks_5",
    "blocks_5_norm1",
    "blocks_5_attn",
    "blocks_5_attn_qkv",
    "blocks_5_attn_q_norm",
    "blocks_5_attn_k_norm",
    "blocks_5_attn_attn_drop",
    "blocks_5_attn_proj",
    "blocks_5_attn_proj_drop",
    "blocks_5_ls1",
    "blocks_5_drop_path1",
    "blocks_5_norm2",
    "blocks_5_mlp",
    "blocks_5_mlp_fc1",
    "blocks_5_mlp_act",
    "blocks_5_mlp_drop1",
    "blocks_5_mlp_norm",
    "blocks_5_mlp_fc2",
    "blocks_5_mlp_drop2",
    "blocks_5_ls2",
    "blocks_5_drop_path2",
    "blocks_6",
    "blocks_6_norm1",
    "blocks_6_attn",
    "blocks_6_attn_qkv",
    "blocks_6_attn_q_norm",
    "blocks_6_attn_k_norm",
    "blocks_6_attn_attn_drop",
    "blocks_6_attn_proj",
    "blocks_6_attn_proj_drop",
    "blocks_6_ls1",
    "blocks_6_drop_path1",
    "blocks_6_norm2",
    "blocks_6_mlp",
    "blocks_6_mlp_fc1",
    "blocks_6_mlp_act",
    "blocks_6_mlp_drop1",
    "blocks_6_mlp_norm",
    "blocks_6_mlp_fc2",
    "blocks_6_mlp_drop2",
    "blocks_6_ls2",
    "blocks_6_drop_path2",
    "blocks_7",
    "blocks_7_norm1",
    "blocks_7_attn",
    "blocks_7_attn_qkv",
    "blocks_7_attn_q_norm",
    "blocks_7_attn_k_norm",
    "blocks_7_attn_attn_drop",
    "blocks_7_attn_proj",
    "blocks_7_attn_proj_drop",
    "blocks_7_ls1",
    "blocks_7_drop_path1",
    "blocks_7_norm2",
    "blocks_7_mlp",
    "blocks_7_mlp_fc1",
    "blocks_7_mlp_act",
    "blocks_7_mlp_drop1",
    "blocks_7_mlp_norm",
    "blocks_7_mlp_fc2",
    "blocks_7_mlp_drop2",
    "blocks_7_ls2",
    "blocks_7_drop_path2",
    "blocks_8",
    "blocks_8_norm1",
    "blocks_8_attn",
    "blocks_8_attn_qkv",
    "blocks_8_attn_q_norm",
    "blocks_8_attn_k_norm",
    "blocks_8_attn_attn_drop",
    "blocks_8_attn_proj",
    "blocks_8_attn_proj_drop",
    "blocks_8_ls1",
    "blocks_8_drop_path1",
    "blocks_8_norm2",
    "blocks_8_mlp",
    "blocks_8_mlp_fc1",
    "blocks_8_mlp_act",
    "blocks_8_mlp_drop1",
    "blocks_8_mlp_norm",
    "blocks_8_mlp_fc2",
    "blocks_8_mlp_drop2",
    "blocks_8_ls2",
    "blocks_8_drop_path2",
    "blocks_9",
    "blocks_9_norm1",
    "blocks_9_attn",
    "blocks_9_attn_qkv",
    "blocks_9_attn_q_norm",
    "blocks_9_attn_k_norm",
    "blocks_9_attn_attn_drop",
    "blocks_9_attn_proj",
    "blocks_9_attn_proj_drop",
    "blocks_9_ls1",
    "blocks_9_drop_path1",
    "blocks_9_norm2",
    "blocks_9_mlp",
    "blocks_9_mlp_fc1",
    "blocks_9_mlp_act",
    "blocks_9_mlp_drop1",
    "blocks_9_mlp_norm",
    "blocks_9_mlp_fc2",
    "blocks_9_mlp_drop2",
    "blocks_9_ls2",
    "blocks_9_drop_path2",
    "blocks_10",
    "blocks_10_norm1",
    "blocks_10_attn",
    "blocks_10_attn_qkv",
    "blocks_10_attn_q_norm",
    "blocks_10_attn_k_norm",
    "blocks_10_attn_attn_drop",
    "blocks_10_attn_proj",
    "blocks_10_attn_proj_drop",
    "blocks_10_ls1",
    "blocks_10_drop_path1",
    "blocks_10_norm2",
    "blocks_10_mlp",
    "blocks_10_mlp_fc1",
    "blocks_10_mlp_act",
    "blocks_10_mlp_drop1",
    "blocks_10_mlp_norm",
    "blocks_10_mlp_fc2",
    "blocks_10_mlp_drop2",
    "blocks_10_ls2",
    "blocks_10_drop_path2",
    "blocks_11",
    "blocks_11_norm1",
    "blocks_11_attn",
    "blocks_11_attn_qkv",
    "blocks_11_attn_q_norm",
    "blocks_11_attn_k_norm",
    "blocks_11_attn_attn_drop",
    "blocks_11_attn_proj",
    "blocks_11_attn_proj_drop",
    "blocks_11_ls1",
    "blocks_11_drop_path1",
    "blocks_11_norm2",
    "blocks_11_mlp",
    "blocks_11_mlp_fc1",
    "blocks_11_mlp_act",
    "blocks_11_mlp_drop1",
    "blocks_11_mlp_norm",
    "blocks_11_mlp_fc2",
    "blocks_11_mlp_drop2",
    "blocks_11_ls2",
    "blocks_11_drop_path2",
    "norm",
    "fc_norm",
    "head_drop",
    "head",
]

some_layers = [
    # "patch_embed",
    # "patch_embed_proj",
    # "patch_embed_norm",
    # "pos_drop",
    # "patch_drop",
    # "norm_pre",
    # "blocks",
    # "blocks_0",
    # "blocks_0_norm1",
    # "blocks_0_attn",
    # "blocks_0_attn_qkv",
    # "blocks_0_attn_q_norm",
    # "blocks_0_attn_k_norm",
    "blocks_0_attn_attn_drop",
    "blocks_0_attn_proj",
    "blocks_0_attn_proj_drop",
    "blocks_0_ls1",
    "blocks_0_drop_path1",
    "blocks_0_norm2",
    "blocks_0_mlp",
    "blocks_0_mlp_fc1",
    "blocks_0_mlp_act",
    "blocks_0_mlp_drop1",
    "blocks_0_mlp_norm",
    "blocks_0_mlp_fc2",
    "blocks_0_mlp_drop2",
    "blocks_0_ls2",
    "blocks_0_drop_path2",
    "blocks_1",
    "blocks_1_norm1",
    "blocks_1_attn",
    "blocks_1_attn_qkv",
    "blocks_1_attn_q_norm",
    "blocks_1_attn_k_norm",
    "blocks_1_attn_attn_drop",
    "blocks_1_attn_proj",
    "blocks_1_attn_proj_drop",
    "blocks_1_ls1",
    "blocks_1_drop_path1",
    "blocks_1_norm2",
    "blocks_1_mlp",
    "blocks_1_mlp_fc1",
    "blocks_1_mlp_act",
    "blocks_1_mlp_drop1",
    "blocks_1_mlp_norm",
    "blocks_1_mlp_fc2",
    "blocks_1_mlp_drop2",
    "blocks_1_ls2",
    "blocks_1_drop_path2",
    "blocks_2",
    "blocks_2_norm1",
    "blocks_2_attn",
    "blocks_2_attn_qkv",
    "blocks_2_attn_q_norm",
    "blocks_2_attn_k_norm",
    "blocks_2_attn_attn_drop",
    "blocks_2_attn_proj",
    "blocks_2_attn_proj_drop",
    "blocks_2_ls1",
    "blocks_2_drop_path1",
    "blocks_2_norm2",
    "blocks_2_mlp",
    "blocks_2_mlp_fc1",
    "blocks_2_mlp_act",
    "blocks_2_mlp_drop1",
    "blocks_2_mlp_norm",
    "blocks_2_mlp_fc2",
    "blocks_2_mlp_drop2",
    "blocks_2_ls2",
    "blocks_2_drop_path2",
    "blocks_3",
    "blocks_3_norm1",
    "blocks_3_attn",
    "blocks_3_attn_qkv",
    "blocks_3_attn_q_norm",
    "blocks_3_attn_k_norm",
    "blocks_3_attn_attn_drop",
    "blocks_3_attn_proj",
    "blocks_3_attn_proj_drop",
    "blocks_3_ls1",
    "blocks_3_drop_path1",
    "blocks_3_norm2",
    "blocks_3_mlp",
    "blocks_3_mlp_fc1",
    "blocks_3_mlp_act",
    "blocks_3_mlp_drop1",
    "blocks_3_mlp_norm",
    "blocks_3_mlp_fc2",
    "blocks_3_mlp_drop2",
    "blocks_3_ls2",
    "blocks_3_drop_path2",
    "blocks_4",
    "blocks_4_norm1",
    "blocks_9",
    "blocks_9_norm1",
    "blocks_9_attn",
    "blocks_9_attn_qkv",
    "blocks_9_attn_q_norm",
    "blocks_9_attn_k_norm",
    "blocks_9_attn_attn_drop",
    "blocks_9_attn_proj",
    "blocks_9_attn_proj_drop",
    "blocks_9_ls1",
    "blocks_9_drop_path1",
    "blocks_9_norm2",
    "blocks_9_mlp",
    "blocks_9_mlp_fc1",
    "blocks_9_mlp_act",
    "blocks_9_mlp_drop1",
    "blocks_9_mlp_norm",
    "blocks_9_mlp_fc2",
    "blocks_9_mlp_drop2",
    "blocks_9_ls2",
    "blocks_9_drop_path2",
    "blocks_10",
    "blocks_10_norm1",
    "blocks_10_attn",
    "blocks_10_attn_qkv",
    "blocks_10_attn_q_norm",
    "blocks_10_attn_k_norm",
    "blocks_10_attn_attn_drop",
    "blocks_10_attn_proj",
    "blocks_10_attn_proj_drop",
    "blocks_10_ls1",
    "blocks_10_drop_path1",
    "blocks_10_norm2",
    "blocks_10_mlp",
    "blocks_10_mlp_fc1",
    "blocks_10_mlp_act",
    "blocks_10_mlp_drop1",
    "blocks_10_mlp_norm",
    "blocks_10_mlp_fc2",
    "blocks_10_mlp_drop2",
    "blocks_10_ls2",
    "blocks_10_drop_path2",
    "blocks_11",
    "blocks_11_norm1",
    "blocks_11_attn",
    "blocks_11_attn_qkv",
    "blocks_11_attn_q_norm",
    "blocks_11_attn_k_norm",
    "blocks_11_attn_attn_drop",
    "blocks_11_attn_proj",
    "blocks_11_attn_proj_drop",
    "blocks_11_ls1",
    "blocks_11_drop_path1",
    "blocks_11_norm2",
    "blocks_11_mlp",
    "blocks_11_mlp_fc1",
    "blocks_11_mlp_act",
    "blocks_11_mlp_drop1",
    "blocks_11_mlp_norm",
    "blocks_11_mlp_fc2",
    "blocks_11_mlp_drop2",
    "blocks_11_ls2",
    "blocks_11_drop_path2",
    "norm",
    "fc_norm",
    "head_drop",
]
