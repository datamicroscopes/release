#!/usr/bin/env python

import sys
import os
import sh

from argparse import ArgumentParser


# Initialize
try:
    token = os.environ['ANACONDA_TOKEN']
except KeyError:
    sys.exit("Must set $ANACONDA_TOKEN")
binstar = sh.Command('binstar').bake(t=token)
conda = sh.Command('conda')


def build_and_publish(path, channel="main"):
    binfile = conda.build("--output", path).strip()
    print conda.build(path).ran
    print binstar.upload(binfile, force=True, channel=channel).ran


def conda_paths(project_name):
    conda_recipes_dir = os.path.join(project_name, 'conda')

    if not os.path.isdir(conda_recipes_dir):
        sys.exit('no such dir: {}'.format(conda_recipes_dir))

    for name in sorted(os.listdir(conda_recipes_dir)):
        yield os.path.join(conda_recipes_dir, name)


def main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--project', required=True)
    parser.add_argument('-s', '--site', required=False, default=None)
    args = parser.parse_args()

    for conda_path in conda_paths(args.project):
        build_and_publish(conda_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
