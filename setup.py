from setuptools import setup

setup(name='hiersum',
      version='0.1',
      description='Extractive summarization of multiple documents',
      url='http://github.com/alingupta/hiersum',
      author='Alind Gupta',
      license='MIT',
      packages=['autosum'],
      install_requires=[
          'numpy',
          'nltk'
      ],
      include_package_data=True)

