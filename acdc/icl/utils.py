from functools import partial
import torch
from acdc.docstring.utils import AllDataThings
import torch.nn.functional as F

import os
import sys

import matplotlib.pyplot as plt
import torch.nn as nn
from fancy_einsum import einsum

from samplers import get_data_sampler
from tasks import get_task_sampler

from munch import Munch
import yaml

class PassThroughEmbed(nn.Module):
        def __init__(self, cfg=None):
            super().__init__()
            # No parameters needed, but constructor accepts cfg for compatibility

        def forward(self, tokens):
            # Directly return the input without any modifications
            return tokens


def get_model(path, device="cpu"):

    model = torch.load(path, map_location=device)
    tl_model = model.to(device)
    tl_model.set_use_attn_result(True)
    tl_model.set_use_split_qkv_input(True)
    if "use_hook_mlp_in" in tl_model.cfg.to_dict():
        tl_model.set_use_hook_mlp_in(True)
    return tl_model

def validation_metric(predictions, labels, return_one_element, device):
    predictions = predictions.to(device)

    sliced_preds = predictions[:, ::2, 0][:, torch.arange(labels.shape[1])]

    loss = (labels - sliced_preds).square().detach().numpy().mean(axis=0)[-5:].mean()
    return loss


def generate_data(conf, read_in_weight, read_in_bias, max_len):
    # generate random data (20d points on a gaussian)

    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size

    data_sampler = get_data_sampler(conf.training.data, n_dims)
    task_sampler = get_task_sampler(
        conf.training.task,
        n_dims,
        batch_size,
        **conf.training.task_kwargs
    )
    task = task_sampler()
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end) # should be n_points=conf.training.curriculum.points.end, but has been hacked to work for the max_len of 101 (202)
    ys = task.evaluate(xs)


    # the original model first merges the sequences in z, which we do here (z can have all but the last y since it's causal and intermediate ys are for icl and later icl lenght eval

    batch, n_ctx, d_xs = xs.shape

    ys_wide = torch.cat(
        (
            ys.view(batch, n_ctx, 1),
            torch.zeros(batch, n_ctx, d_xs - 1, device=ys.device),
        ),
        axis=2,
    )
    my_zs = torch.stack((xs, ys_wide), dim=2)
    my_zs = my_zs.view(batch, 2 * n_ctx, d_xs)

    # apply the read_in transformation
    transformed_zs = einsum("batch n_ctx d_xs, d_model d_xs -> batch n_ctx d_model", my_zs, read_in_weight) + read_in_bias

    # apply padding

    current_len = transformed_zs.shape[1]

    pad_len = max(max_len - current_len, 0)

    # Apply padding to the right of the second dimension
    # The padding order in F.pad is (left, right, top, bottom) for 4D input, but here it's the equivalent for 3D
    return F.pad(transformed_zs, (0, 0, 0, pad_len), "constant", 0), ys

def get_conf():
    run_dir = "models"

    task = "linear_regression"
    run_id = "pretrained"  # if you train more models, replace with the run_id from the table above
    run_path = os.path.join(run_dir, task, run_id)
    config_path = os.path.join(run_path, "config.yaml")

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    
    return conf
        


def get_all_icl_things(device='cpu', return_one_element=False) -> AllDataThings:

    

    conf = get_conf()

    model = get_model('hooked_regressor.pt', device)
    read_in_weight = torch.load('read_in_weight.pt', map_location=device)
    read_in_bias = torch.load('read_in_bias.pt', map_location=device)

    validation_data, validation_correct_answers = generate_data(conf, read_in_weight, read_in_bias, model.cfg.n_ctx)
    validation_patch_data, _ = generate_data(conf, read_in_weight, read_in_bias, model.cfg.n_ctx)

    test_data, test_correct_answers = generate_data(conf, read_in_weight, read_in_bias, model.cfg.n_ctx)
    test_patch_data, _ = generate_data(conf, read_in_weight, read_in_bias, model.cfg.n_ctx)



    return AllDataThings(
        tl_model=model,
        validation_metric=partial(validation_metric, correct=validation_correct_answers, return_one_element=return_one_element, device=device),
        validation_data=validation_data,
        validation_labels=validation_correct_answers,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=partial(validation_metric, correct=test_correct_answers, return_one_element=return_one_element, device=device),
        test_data=test_data,
        test_labels=test_correct_answers,
        test_mask=None,
        test_patch_data=test_patch_data,
    )


# Testing (pay attention, this is now buggd and would require valid function outside of get things):

import unittest
class TestICLDataAndModel(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'
        self.model = get_model('hooked_regressor.pt')
        self.read_in_weight = torch.load('read_in_weight.pt', map_location='cpu')
        self.read_in_bias = torch.load('read_in_bias.pt', map_location='cpu')
        self.conf = get_conf()

    def test_validation_metric(self):
        data, correct_answers = generate_data(self.conf, self.read_in_weight, self.read_in_bias, self.model.cfg.n_ctx)
        # Assuming validation_metric is defined elsewhere in your code
        preds = self.model(data)
        mse = validation_metric(predictions=preds, labels=correct_answers, return_one_element=False, device=self.device)
        print('This is the MSE: ', mse)

# Execute tests when the script is run
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
