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
        type=str,
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
        inference_mse = run.data.metrics.get("inference_mse")
        inference_mae = run.data.metrics.get("inference_mae")

        if (
            not isinstance(time_start, str)
            or not isinstance(time_end, str)
            or not isinstance(train_95_percentile, str)
        ):
            continue

        try:
            time_start = datetime.fromisoformat(time_start)
            time_end = datetime.fromisoformat(time_end)
            train_95_percentile = float(train_95_percentile)
        except Exception as e:
            print(f"{run.info.run_id}\n{e}")
            continue
        if not isinstance(inference_mae, float) or not isinstance(inference_mse, float):
            if train_95_percentile < 0.0001:
                inference_mae = float(0.0)
                inference_mse = float(0.0)
            else:
                raise RuntimeError(
                    f"Bad metrics exp: {experiment_id}, run {run.info.run_id}"
                )

        date_res = (
            f"{time_start.month}.{time_start.day}-{time_end.month}.{time_end.day}"
        )
        runs2mae.update({date_res: round(inference_mae, 4)})
        runs2mse.update({date_res: round(inference_mse, 4)})

    return runs2mae, runs2mse
