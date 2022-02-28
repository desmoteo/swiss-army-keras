from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='swiss_army_keras',
      version='0.4.0',
      description='A collection of models and utilities for the development of edge deployable Keras models',
      long_description=long_description,
      keywords=['keras', 'segmentation', 'classification', 'finetuning', 'edge',
                'quantization', 'augmentation'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Multimedia :: Graphics'],
      url='http://github.com/desmoteo/swiss-army-keras',
      author='Matteo Ferrabone',
      author_email='matteo.ferrabone@gmail.com',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      package_data={
          # If any package contains *.txt files, include them:
          "": ["*.ttf"],
      },
      install_requires=[
          'numpy',
          #'tensorflow>=2.4.0',
          #'efficientnet-lite-keras @ git+https://github.com/waterviewsrl/efficientnet-lite-keras.git',
          'irondomo',
          'pillow',
          'typeguard'
      ],
      zip_safe=False)
