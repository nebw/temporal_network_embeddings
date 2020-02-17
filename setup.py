#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]


setup(
    name='temporal_network_embeddings',
    version='0.1',
    description='Learning of temporal network dynamics using recurrent neural networks',
    author='Benjamin Wild',
    author_email='b.w@fu-berlin.de',
    url='https://github.com/nebw/temporal_network_embeddings/',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['tne'],
)
