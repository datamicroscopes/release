#!/usr/bin/env python

import sys
import socket
import os

from argparse import ArgumentParser
from subprocess import check_call, check_output


def ensure_tool(name):
    check_call(['which', name])


def build_and_publish(path, args):
    login_command = get_login_command(args)
    print >>sys.stderr, "Test anaconda.org login:"
    check_call(login_command)

    binfile = check_output(['conda', 'build', '--output', path])
    binfile = binfile.strip()
    print >>sys.stderr, "build path {}".format(binfile)

    print >>sys.stderr, "conda build {}".format(path)
    check_call(['conda', 'build', path])

    upload_command = "binstar upload --force {}".format(binfile)

    login_and_upload_command = "{} && {}".format(login_command, upload_command)
    print >>sys.stderr, "Login to binstar and upload"
    check_call(login_and_upload_command)


def get_login_command(args):
    return ("binstar login --hostname {hostname} "
            " --username {username} --password {password}")\
        .format(
        username=args.username,
        password=args.password,
    )


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
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-P', '--password', required=True)
    parser.add_argument('-p', '--project', required=True)
    parser.add_argument('-s', '--site', required=False, default=None)
    args = parser.parse_args()

    # make sure we have a conda environment
    ensure_tool('conda')
    ensure_tool('binstar')

    conda_recipes_dir = get_conda_recipes_dir(args.project)

    for conda_path in conda_paths(conda_recipes_dir):
        build_and_publish(conda_path, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
