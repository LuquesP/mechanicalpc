from networks.pct.pct import Point_Transformer


def pct_get_model():
    config = {
        "num_points": 2048,
        "batch_size": 64,
        "use_normals": False,
        "optimizer": "Adam",
        "lr": 0.001,
        "decay_rate": 1e-06,
        "epochs": 30,
        "num_classes": 62,
        "dropout": 0.4,
        "M": 4,
        "K": 64,
        "d_m": 512,
    }
    model = Point_Transformer(config)
    return model
