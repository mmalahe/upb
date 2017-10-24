from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))

setup(name='upb',
      packages=[package for package in find_packages()
                if package.startswith('upb')],
      install_requires=[
          'baselines',
          'numpy',
          'py_mini_racer',
          'selenium',
          'lxml',
          'gym',
          'matplotlib',
          'tensorflow >= 1.0.0',
      ],
      description="Universal Paperclips Bot: a bot that plays Universal Paperclips",
      author="Michael Malahe",
      url='https://github.com/mmalahe/upb',
      author_email="",
      version="0.0.2")
