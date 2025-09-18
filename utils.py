from argparse import ArgumentParser
from datetime import datetime

from mlflow import MlflowClient


def gen_parser():
    parser: ArgumentParser = ArgumentParser(prog="main.py")
    _ = parser.add_argument(
        "-n",
        "--subsystems",
        type=str,
        required=True,
        default="./input",
        help="путь до файла с списком подсистем на анализ",
    )
    _ = parser.add_argument(
        "-u",
        "--tracking-uri",
        type=str,
        required=True,
        help="адрес сервера mlflow",
    )
    _ = parser.add_argument(
        "-e",
        "--experiment-id",
        type=int,
        required=True,
        help="id эксперимента для анализа",
    )
    return parser


def get_subsystem_metrics(client: MlflowClient, experiment_id: str, name: str):
    filter_string = f'attributes.run_name = "{name}" AND attributes.status = "FINISHED"'
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["tags.version DESC"],
    )

    runs2mae: dict[str, float] = {}
    runs2mse: dict[str, float] = {}

    for run in runs[:-2]:
        time_start = run.data.params.get("inference_date_time_start")
        time_end = run.data.params.get("inference_date_time_end")
        train_95_percentile = run.data.params.get("train_95_percentile")
        inference_mse = run.data.params.get("inference_mse")
        inference_mae = run.data.params.get("inference_mae")

        if (
            not isinstance(time_start, str)
            or not isinstance(time_end, str)
            or not isinstance(train_95_percentile, str)
            or not isinstance(inference_mse, str)
            or not isinstance(inference_mae, str)
        ):
            continue

        try:
            time_start = datetime.fromisoformat(time_start)
            time_end = datetime.fromisoformat(time_end)
            train_95_percentile = float(train_95_percentile)
            inference_mae = float(inference_mae)
            inference_mse = float(inference_mse)
        except Exception as e:
            print(f"{run.info.run_name}\n{e}")
            continue

        date_res = (
            f"{time_start.month}.{time_start.day}-{time_end.month}.{time_end.day}"
        )
        runs2mae.update({date_res: inference_mae})
        runs2mse.update({date_res: inference_mse})

    return runs2mae, runs2mse
