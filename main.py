from argparse import Namespace
import json
import pandas as pd

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

    mae_index: list[str] = []
    mse_index: list[str] = []
    mae_datas: dict[str, list[float]] = {}
    mse_datas: dict[str, list[float]] = {}

    for subsystem in file_list:
        mae_result, mse_result = get_subsystem_metrics(
            client=client, experiment_id=args.experiment_id, name=subsystem
        )
        mae_index = list(mae_result.keys())
        mae_datas.update({subsystem: list(mae_result.values())})
        mse_index = list(mse_result.keys())
        mse_datas.update({subsystem: list(mse_result.values())})

    mae_pd = pd.DataFrame(data=mae_datas, index=pd.Index(mae_index))
    mse_pd = pd.DataFrame(data=mse_datas, index=pd.Index(mse_index))

    # Запись на разные листы Excel
    with pd.ExcelWriter(f"./{args.experiment_id}.xlsx", engine="openpyxl") as writer:
        mae_pd.to_excel(writer, sheet_name="MAE", index=True)
        mse_pd.to_excel(writer, sheet_name="MSE", index=True)


if __name__ == "__main__":
    parser = gen_parser()
    main(parser.parse_args())
