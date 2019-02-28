#!/usr/bin/env python3

def main():
    try:
        from setuptools import setup, find_packages
    except ImportError:
        from distutils.core import setup

    options = {
        'name': 'ising_kitaev',
        'version': '0.0.1',
        'license': 'MIT license',
        'description': "",
        'long_description': "",
        'packages': ['ising_kitaev'],
    }
    setup(**options)


if __name__ == '__main__':
    main()
