import functools

from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy,size_based_auto_wrap_policy


def get_wrapper(cfg,block=None):
    if cfg.wrapper_type == "size_based":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=cfg.min_num_params,
        )
    elif cfg.wrapper_type == "transformer":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                block,
            },
        )

    return auto_wrap_policy
