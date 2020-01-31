import os
import json
import argparse

import settings

import logging
import logging.config

from run_network import run

def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args([])

    with open('config.json', 'r') as f:
        config = json.load(f)
    
    for key, item in config['WGAN_GP'].items():
        setattr(args, key, item)

    run(args)

if __name__ == "__main__":
    main()