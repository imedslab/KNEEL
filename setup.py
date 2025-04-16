from setuptools import setup, find_packages

setup(
    name='kneel',
    version='0.2',
    author='Aleksei Tiulpin',
    author_email='aleksei.tiulpin@oulu.fi',
    packages=find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    include_package_data=True,
    license='Creative Commons Attribution Non Commercial 4.0',
    long_description=open('README.md').read(),
)