import sys
import sh

from fabric import api as fab

sed = sh.sed.bake('-i bak -e')
TRAVIS_YAML = '.travis.yml'
REPLACE_LANGUAGE = 's/language: .*/language: {}/'
REPLACE_CHANNEL = 's/ANACONDA_CHANNEL=.*/ANACONDA_CHANNEL={}/'


def is_dirty():
    return "" != sh.git.status(porcelain=True).strip()


def _release(language, message, channel):
    if is_dirty():
        sys.exit("Repo must be in clean state before deploying. Please commit changes.")
    sed(REPLACE_LANGUAGE.format(language), TRAVIS_YAML)
    sed(REPLACE_CHANNEL.format(channel), TRAVIS_YAML)
    if is_dirty():
        sh.git.add(TRAVIS_YAML)
    # sh.git.commit(m=message, allow_empty=True)
    # sh.git.pull(rebase=True)
    # sh.git.push()


@fab.task
def update():
    if is_dirty():
        sys.exit("Repo must be in clean state before deploying. Please commit changes.")
    sh.git.submodule.update(remote=True, rebase=True)
    if is_dirty():
        print "Updated repositories:"
        print sh.git.status(porcelain=True).strip()
        sh.git.add(all=True)
        sh.git.commit(m="Update submodules to origin")
    else:
        sys.exit('Nothing to update.')


@fab.task
def release(os, channel="main"):
    if os.lower() == "osx":
        _release('objective-c', "Release OS X", channel)
    elif os.lower() == "linux":
        _release('python', "Release Linux", channel)
    else:
        sys.exit("Specify either osx or linux.")


@fab.task
def release_osx(channel="main"):
    release("osx", channel)


@fab.task
def release_linux(channel="main"):
    release("linux", channel)
