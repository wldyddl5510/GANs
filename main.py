import os
import json
import argparse

import settings

import logging
import logging.config

from run_network import run

def main():
    tmp_parser = argparse.ArgumentParser(
        description="temporary parser to figure out which configuration to work on"
    )

    # Get configuration
    tmp_parser.add_argument("--config_name", type = str, default = "HIGH_WGAN")
    config_name = tmp_parser.parse_args().config_name

    # open config file
    with open('config.json', 'r') as f:
        config = json.load(f)

    # parse argument
    parser = argparse.ArgumentParser(
        description = "higher dimension reconstruction gan"
    )
    args = parser.parse_args([])
    for key, item in config[config_name].items():
        setattr(args, key, item)

    run(args)

if __name__ == "__main__":
    main()