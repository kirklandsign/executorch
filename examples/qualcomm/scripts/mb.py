# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np

import torch
from executorch.examples.models.mobilebert import MobileBertModelExample
from executorch.examples.qualcomm.scripts.utils import (
    build_executorch_binary,
    make_output_dir,
    SimpleADB,
    topk_accuracy,
)

import sys
sys.setrecursionlimit(50000)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--artifact",
        help="path for storing generated artifacts by this example. "
        "Default ./mb",
        default="./mb",
        type=str,
    )
    parser.add_argument(
        "-b",
        "--build_folder",
        help="path to cmake binary directory for android, e.g., /path/to/build_android",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--device",
        help="serial number for android device communicated via ADB.",
        type=str,
        required=False,
    )
    parser.add_argument(
        "-H",
        "--host",
        help="hostname where android device is connected.",
        default=None,
        type=str,
    )
    parser.add_argument(
        "-m",
        "--model",
        help="SoC model of current device. e.g. 'SM8550' for Snapdragon 8 Gen 2",
        type=str,
        required=True,
    )

    # QNN_SDK_ROOT might also be an argument, but it is used in various places.
    # So maybe it's fine to just use the environment.
    if "QNN_SDK_ROOT" not in os.environ:
        raise RuntimeError("Environment variable QNN_SDK_ROOT must be set")
    print(f"QNN_SDK_ROOT={os.getenv('QNN_SDK_ROOT')}")

    if "LD_LIBRARY_PATH" not in os.environ:
        print(
            "[Warning] LD_LIBRARY_PATH is not set. If errors like libQnnHtp.so "
            "not found happen, please follow setup.md to set environment."
        )
    else:
        print(f"LD_LIBRARY_PATH={os.getenv('LD_LIBRARY_PATH')}")

    args = parser.parse_args()

    # ensure the working directory exist.
    os.makedirs(args.artifact, exist_ok=True)

    inputs = (torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 10140, 102]]),)
    pte_filename = "mb_qnn"
    instance = MobileBertModelExample()
    build_executorch_binary(
        instance.get_eager_model().eval(),
        inputs, # instance.get_example_inputs(),
        args.model,
        f"{args.artifact}/{pte_filename}",
        inputs,
    )
