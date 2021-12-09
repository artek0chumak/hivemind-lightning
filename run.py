import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace, REMAINDER
from typing import Any

import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def get_args_parser() -> ArgumentParser:
    """Helper function parsing the command line options."""

    parser = ArgumentParser(description="Lightning HiveMind")
    parser.add_argument(
        "--num_local_clients",
        type=int,
        default=2,
        help="Number of local clients to run on the local machine.",
    )
    parser.add_argument(
        "--coordinator_address",
        type=str,
        default="127.0.0.1:8000",
        help="The address of the coordinating server for using hivemind.",
    )
    parser.add_argument(
        "--training_name",
        type=str,
        default="pl_training",
        help="The address of the host server. Default is pl_training. Must be unique!",
    )
    parser.add_argument(
        "--disable_client_logging",
        action="store_true",
        default=False,
        help="Disable client logging to client file",
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="Full path to the training program/script to be launched in parallel, "
             "followed by all the arguments for the training script.",
    )
    parser.add_argument(
        "--gpu_per_client",
        action="store_true",
        default=False,
        help="Assigns each client a local GPU. "
             "If there are more clients than GPUs, evenly distributes clients across all GPUs.",
    )

    # Rest from the training program.
    parser.add_argument("training_script_args", nargs=REMAINDER)
    return parser


def parse_args(args: Any = None) -> Namespace:
    parser = get_args_parser()
    return parser.parse_args(args)


def _start_server(args: Namespace) -> subprocess.Popen:
    command = [sys.executable] + [args.training_script] + args.training_script_args
    return subprocess.Popen(command, env=os.environ.copy())


def main() -> None:
    args = parse_args()

    if args.coordinator_address:
        import requests

        r = requests.get(f"http://{args.coordinator_address}/get-dht/{args.training_name}")
        if r.status_code != 200:
            raise RuntimeError("could not get DHT address")
        resp = r.json()
        os.environ["INITIAL_PEERS"] = ",".join(resp["peers"])

    if args.gpu_per_client and not torch.cuda.is_available():
        raise MisconfigurationException("Torch was not able to discover any local GPUs.")

    num_gpus = torch.cuda.device_count() if args.gpu_per_client else 0

    num_clients = args.num_local_clients

    os.environ["WORLD_SIZE"] = str(num_clients)

    client_procs = []
    for client_rank in range(num_clients):
        env_copy = os.environ.copy()
        if num_gpus > 0:
            # choose single GPU device rank in round robin fashion
            env_copy["CUDA_VISIBLE_DEVICES"] = str(client_rank % num_gpus)
        env_copy["HIVEMIND_RANK"] = str(client_rank)
        command = [sys.executable] + [args.training_script] + args.training_script_args
        print("STARTING PROC", command, client_rank)
        if client_rank != 0:
            std = subprocess.DEVNULL if args.disable_client_logging else open(f"{client_rank}.log", "w")
            proc = subprocess.Popen(command, env=env_copy, stdout=std, stderr=std)  # type: ignore
        else:
            proc = subprocess.Popen(command, env=env_copy)  # type: ignore
        client_procs.append(proc)
    for client_proc in client_procs:
        client_proc.wait()


if __name__ == '__main__':
    main()
