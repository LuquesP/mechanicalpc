from dataclasses import dataclass


@dataclass
class repsurf:
    model: str
    cuda_ops: bool
    num_point: int
    return_dist: bool
    return_center: bool
    return_polar: bool
    group_size: int
    umb_pool: str
    num_class: int
    normal: bool


def get_repsurf_args():
    args = repsurf(
        "repsurf_ssg_umb", False, 2048, True, True, True, 8, "sum", 68, False
    )
    return args
