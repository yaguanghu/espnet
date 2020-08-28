#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch


import numpy as np


def main():

    last = sorted(args.snapshots, key=os.path.getmtime)
    last = last[-args.num :]
    print("average over", last)
    avg = None

    # sum
    for path in last:
        states = torch.load(path, map_location=torch.device("cpu"))
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]

    # average
    for k in avg.keys():
        if avg[k] is not None and not str(avg[k].dtype).startswith("torch.int"):
            avg[k] /= args.num

    torch.save(avg, args.out)



def get_parser():
    parser = argparse.ArgumentParser(description="average models from snapshot")
    parser.add_argument("--snapshots", required=True, type=str, nargs="+")
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--num", default=10, type=int)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main()
