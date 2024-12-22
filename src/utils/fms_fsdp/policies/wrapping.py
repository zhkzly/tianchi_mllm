import functools

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


def get_wrapper(train_args, block=None):
    if train_args.wrapper_type == "size_based":
        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=train_args.min_num_params,
        )
    # 可以通过自己导入的形式，选择合适的层进行block
    elif train_args.wrapper_type == "transformer":
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                block,
            },
        )

    return auto_wrap_policy
