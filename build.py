#!/usr/bin/env python

import sys
import os

from argparse import ArgumentParser
from subprocess import check_call, check_output


def ensure_tool(name):
    check_call(['which', name])


def build_and_publish(path, token):

    binfile = check_output(['conda', 'build', '--output', path])
    binfile = binfile.strip()
    print >>sys.stderr, "build path {}".format(binfile)

    print >>sys.stderr, "conda build {}".format(path)
    check_call(['conda', 'build', path])

    upload_command = "binstar -t {} upload --force {}".format(token, binfile)

    print >>sys.stderr, "Upload to Anaconda.org"
    check_call(upload_command, shell=True)


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

    # make sure we have a conda environment
    ensure_tool('conda')
    ensure_tool('binstar')

    conda_recipes_dir = get_conda_recipes_dir(args.project)

    for conda_path in conda_paths(conda_recipes_dir):
        build_and_publish(conda_path, args.token)
    return 0


if __name__ == '__main__':
    sys.exit(main())
