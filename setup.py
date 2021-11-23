import codecs
import os
import re
from setuptools import setup


# cf https://packaging.python.org/guides/single-sourcing-package-version/
def read(*parts):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def get_required():
    file = read('requirements.txt')
    return file.split('\n')


def get_docs_extras():
    file = read('docs', 'requirements.txt')
    return file.split('\n')


def get_long_description():
    return read('README.md')


setup(name='tutti',
      version=find_version('tutti', '__init__.py'),
      description='A simple portfolio optimiser beyond the mean-variance optimisation',
      long_description=get_long_description(),
      long_description_content_type='text/markdown',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Office/Business :: Financial :: Investment',
      ],
      keywords='finance investment optimisation',
      url='https://github.com/thoriuchi0531/tutti',
      author='thoriuchi0531',
      author_email='thoriuchi0531@gmail.com',
      license='MIT',
      packages=['tutti'],
      install_requires=get_required(),
      extras_require={
          'dev': [
                     'pytest',
                     'coverage',
                 ] + get_docs_extras(),
      },
      zip_safe=False,
      include_package_data=True,
      python_requires='>=3.6',
      )
