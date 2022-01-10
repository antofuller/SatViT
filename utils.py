import torch
import math


def adjust_learning_rate(optimizer, epoch, sched_config):
    """Decay the learning rate with half-cycle cosine after warmup
    sched_config['lr'] -> maximum learning rate
    sched_config['epochs'] -> epochs we will train for
    epoch -> the current epoch, or iteration (between 0 and sched_config['epochs'])
    Note: The learning rate becomes cyclic only if epoch > sched_config['epochs']
    """
    if epoch < sched_config['warmup_epochs']:
        lr = sched_config['lr'] * epoch / sched_config['warmup_epochs']

    else:
        lr = sched_config['min_lr'] + (sched_config['lr'] - sched_config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - sched_config['warmup_epochs']) / (sched_config['epochs'] - sched_config['warmup_epochs'])))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

