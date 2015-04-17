#!/usr/bin/env python

import sys
import os
import socket

from argparse import ArgumentParser
from subprocess import check_call, check_output
from conda import config as cc
from conda_build.metadata import MetaData
from binstar_client import NotFound
from binstar_client.utils import get_config, get_binstar, store_token

def ensure_tool(name):
    check_call(['which', name])

def build_and_publish(path, binstar, username, remove_before_build):
    binfile = check_output(['conda', 'build', '--output', path])
    binfile = binfile.strip()

    if remove_before_build:
        # XXX(stephentu): figure out if this is needed
        #
        # check to see if this particular distribution already exists in
        # binstar.  If so, remove it first. Why? I am not entirely sure if
        # conda is actually using the *new* version we are building (instead of
        # the one currently located in binstar) when running conda tests!
        # if we remove it first from binstar, then we guarantee it must be using
        # the newly built version
        #
        # this code mostly comes from:
        #    conda_build/build.py
        #    binstar_client/commands/upload.py
        m = MetaData(path)
        basefilename = os.path.join(cc.subdir, "{}.tar.bz2".format(m.dist()))
        package_name = m.meta['package']['name']
        version = m.meta['package']['version']
        args = [username, package_name, version, basefilename]
        print >>sys.stderr, "looking for existing distribution:", args
        try:
            info = binstar.distribution(*args)
            print >>sys.stderr, "removing existing distribution: {}".format(info)
            binstar.remove_dist(*args)
        except NotFound:
            print >>sys.stderr, "no existing distribution found"

    print >>sys.stderr, "conda build {}".format(path)
    check_call(['conda', 'build', path])

    print >>sys.stderr, "binstar upload --force {}".format(binfile)
    check_call(['binstar', 'upload', '--force', binfile])

def main():
    parser = ArgumentParser()
    parser.add_argument('-u', '--username', required=True)
    parser.add_argument('-P', '--password', required=True)
    parser.add_argument('-p', '--project', required=True)
    parser.add_argument('-s', '--site', required=False, default=None)
    parser.add_argument('--remove-before-build', action='store_true')
    args = parser.parse_args()

    # make sure we have a conda environment
    ensure_tool('conda')
    ensure_tool('binstar')

    # make sure the project has a conda recipes folder
    conda_recipes_dir = os.path.join(args.project, 'conda')
    if not os.path.isdir(conda_recipes_dir):
        print >>sys.stderr, 'no such dir: {}'.format(conda_recipes_dir)
        return 1

    # login to binstar
    # this code is taken from:
    #   binstar_client/commands/login.py
    bs = get_binstar()
    config = get_config()
    url = config.get('url', 'https://api.binstar.org')
    token = bs.authenticate(
        args.username, args.password,
        'binstar_client:{}'.format(socket.gethostname()),
        url, created_with='')
    if token is None:
        print >>sys.stderr, 'could not login'
        return 1
    store_token(token, args)

    for name in sorted(os.listdir(conda_recipes_dir)):
        build_and_publish(
            os.path.join(conda_recipes_dir, name),
            bs,
            args.username,
            args.remove_before_build)

    return 0

if __name__ == '__main__':
    sys.exit(main())
