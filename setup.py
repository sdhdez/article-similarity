# -*- coding: utf-8 -*-
""" setup

Package setup for kpext

"""

from setuptools import setup, find_packages

with open('README.md') as f:
    README = f.read()

setup(name='somhos',
      version='v0.1.0.dev0',
      description='Document similarity',
      long_description=README,
      long_description_content_type='text/markdown',
      url='https://github.com/sdhdez/article-similarity.git',
      author='Simon D. Hernandez',
      author_email='py.somhos@totum.one',
      license='MIT',
      classifiers=[
          'Development Status :: 1 - Planning',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
      ],
      keywords='',
      packages=find_packages('src'),
      package_dir={'':'src', 'somhos': 'src/somhos'},
      package_data={'somhos': []},
      python_requires='>=3.5, <4',
      platform='any',
      install_requires=['kleis-keyphrase-extraction', 'whoosh', 'tensorflow', 'scipy', 'gensim', 'jupyterlab'],
      zip_safe=False)
