from pyexpat import features


def main():
    spec = parse_args()

    model = load_model(spec["model"])

    data = load_dataset(
        dataset_name=spec["dataset"],
        features=model.features,
        robot_type=spec["robot"],
    )
    print(spec)

if __name__ == "__main__":
    from parse_args import parse_args
    from data_utils import load_dataset
    from models import load_model

    main()
else:
    from .parse_args import parse_args
