def plot_preds_vs_gt(model, dataloader):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    total_y_true = []
    total_y_pred = []
    total_ts = []

    with torch.no_grad():
        for x, y, t in dataloader:
            y_pred = model(x)
            y_pred *= t
            total_y_true.extend(y.numpy())
            total_y_pred.extend(y_pred.numpy())
            total_ts.extend(t.numpy())

    np.save("data/y_pred.npy", np.array(total_y_pred))
    np.save("data/y_true.npy", np.array(total_y_true))
    np.save("data/ts.npy", np.array(total_ts))

    total_y_pred = dataloader.dataset.rollout_single_seq(np.array(total_y_pred))
    total_y_true = dataloader.dataset.rollout_single_seq(np.array(total_y_true))

    # total_y_pred = np.cumsum(total_y_pred, axis=1)
    # total_y_true = np.cumsum(total_y_true, axis=1)

    plt.plot(total_y_pred[:, 0], total_y_pred[:, 1], label="Pred")
    plt.plot(total_y_true[:, 0], total_y_true[:, 1], label="True")
    plt.legend()
    plt.show()


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
        from torch.utils.data import DataLoader
        dl = DataLoader(data, batch_size=32, shuffle=False)
        model = model()

        train_torch_simple(model, optim.Adam(model.parameters()), dl, nn.MSELoss(), 20)

        plot_preds_vs_gt(model, dl)



if __name__ == "__main__":
    from parse_args import parse_args
    from data_utils import load_dataset
    from models import load_model
    from optimization_logic import train_numpy, train_torch_simple

    main()
