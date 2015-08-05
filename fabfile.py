import sys
import sh

from fabric import api as fab

sed = sh.sed.bake('-i bak -e')
TRAVIS_YAML = '.travis.yml'
REPLACE_LANGUAGE = 's/language: .*/language: {}/'


def is_dirty():
    return "" != sh.git.status(porcelain=True).strip()


def release(language, message):
    if is_dirty():
        sys.exit("Repo must be in clean state before deploying. Please commit changes.")
    sed(REPLACE_LANGUAGE.format(language), TRAVIS_YAML)
    if is_dirty():
        sh.git.add(TRAVIS_YAML)
    sh.git.commit(m=message, allow_empty=True)
    sh.git.pull(rebase=True)
    sh.git.push()



@fab.task
def release_osx():
    release('objective-c', "Release OS X")


@fab.task
def release_linux():
    release('python', "Release Linux")
