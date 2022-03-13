def main():
    spec = parse_args()

    model = load_model(spec["model"])

    data = load_dataset(
        dataset_name=spec["dataset"],
        x_features=model.x_features,
        y_features=model.y_features,
        robot_type=spec["robot"],
        dataset_type=model.dataset_name,
    )

    if model.dataset_name == "numpy":
        model = train_numpy(data, model)
    elif model.dataset_name == "torch_lookahead":
        from torch import optim, nn
        model = model()
        train_torch_simple(model, optim.Adam(model.parameters()), data, nn.MSELoss(), 100)


if __name__ == "__main__":
    from parse_args import parse_args
    from data_utils import load_dataset
    from models import load_model
    from optimization_logic import train_numpy, train_torch_simple

    main()
