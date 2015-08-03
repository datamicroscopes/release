#!/usr/bin/env python

import sys
import os
import sh

from argparse import ArgumentParser


binstar = sh.Command('binstar')
conda = sh.Command('conda')


def build_and_publish(path, token):
    binfile = conda.build.bake(output=True)(path).strip()
    conda.build(path)
    binstar.bake(t=token).upload(binfile, force=True)


def get_conda_recipes_dir(project):
    # make sure the project has a conda recipes folder
    conda_recipes_dir = os.path.join(project, 'conda')
    if not os.path.isdir(conda_recipes_dir):
        sys.exit('no such dir: {}'.format(conda_recipes_dir))
    return conda_recipes_dir


def conda_paths(conda_recipes_dir):
    for name in sorted(os.listdir(conda_recipes_dir)):
        yield os.path.join(conda_recipes_dir, name)


def main():
    parser = ArgumentParser()
    parser.add_argument('-u', '--token', required=True)
    parser.add_argument('-p', '--project', required=True)
    parser.add_argument('-s', '--site', required=False, default=None)
    args = parser.parse_args()

    conda_recipes_dir = get_conda_recipes_dir(args.project)

    for conda_path in conda_paths(conda_recipes_dir):
        build_and_publish(conda_path, args.token)
    return 0


if __name__ == '__main__':
    sys.exit(main())
