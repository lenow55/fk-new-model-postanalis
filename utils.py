from argparse import ArgumentParser


def gen_parser():
    parser: ArgumentParser = ArgumentParser(prog="subsystem_map.py")
    _ = parser.add_argument(
        "-n",
        "--new-report",
        type=str,
        required=True,
        default="./input",
        help="путь до файла с метриками по модели",
    )
    _ = parser.add_argument(
        "-p",
        "--models",
        type=str,
        required=True,
        default="./models",
        help="путь до папки с моделями по подсистемам",
    )
    _ = parser.add_argument(
        "-d",
        "--model_id",
        type=int,
        required=True,
        help="id модели для обработки",
    )
    _ = parser.add_argument(
        "-s",
        "--start",
        type=str,
        required=True,
        help="Время, с которого инференс",
    )
    _ = parser.add_argument(
        "-e",
        "--end",
        type=str,
        required=True,
        help="Время, до которого будет идти инференс",
    )
    _ = parser.add_argument(
        "-n",
        "--name",
        type=str,
        required=False,
        default="inference",
        help="название рана на mlflow",
    )
    _ = parser.add_argument(
        "--err_th",
        type=int,
        required=False,
        default=35,
        help="трэшхолд для ошибок",
    )
    _ = parser.add_argument(
        "--req_th",
        type=int,
        required=False,
        default=40,
        help="трэшхолд для запросов",
    )
    _ = parser.add_argument(
        "--err2req_th",
        type=int,
        required=False,
        default=40,
        help="трэшхолд для ошибок на запросов",
    )
    return parser
