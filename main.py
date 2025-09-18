from argparse import Namespace
import json

from mlflow import MlflowClient

from utils import gen_parser, get_subsystem_metrics


def main(args: Namespace):
    if not isinstance(args.subsystems, str):
        raise RuntimeError("Bad subsystems file path")
    if not isinstance(args.tracking_uri, str):
        raise RuntimeError("Bad tracking_uri")
    if not isinstance(args.experiment_id, str):
        raise RuntimeError("Bad experiment_id")

    with open(args.subsystems, "r") as f:
        file_list: list[str] = json.load(f)

    if not isinstance(file_list, list):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise RuntimeError("Bad subsystems file content")  # pyright: ignore[reportUnreachable]

    client = MlflowClient(tracking_uri=args.tracking_uri)

    for subsystem in file_list:
        result = get_subsystem_metrics(
            client=client, experiment_id=args.experiment_id, name=subsystem
        )


if __name__ == "__main__":
    parser = gen_parser()
    main(parser.parse_args())
