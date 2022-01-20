import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--list-models", action="store_true", help="Models to use. If True lists all models and exists.")

    parser.add_argument(
        "-d", "--dataset", type=str, help="Which dataset to use. Should be a folder under datasets.", required='--list-models' not in sys.argv
    )
    parser.add_argument(
        "-r", "--robot", choices=["ackermann", "skid"], help="Type of robot to use.", required='--list-models' not in sys.argv
    )

    parser.add_argument(
        "-m", "--model", type=str, help="Name of model to use. To see which models are available see --list-models.", required='--list-models' not in sys.argv
    )

    result = vars(parser.parse_args()).copy()

    if result["list_models"]:
        from models import get_all_models
        print(
            "Available models are:\n  " + "\n  ".join(get_all_models().keys())
        )
        exit(0)

    return result
