#!/usr/bin/env python3

import sys

from src.marco.marco import parse_args, enumerate_with_args


def main():
    args_list = sys.argv[1:]
    if '--improved-implies' not in args_list:
        args_list = ['--improved-implies'] + args_list
    if '--sat-map-assist' not in args_list:
        args_list = ['--sat-map-assist'] + args_list
    if '--hybrid-shrink' not in args_list:
        args_list = ['--hybrid-shrink'] + args_list
    args = parse_args(args_list)
    for result in enumerate_with_args(args, print_results=True):
        print(result)


if __name__ == '__main__':
    main()
