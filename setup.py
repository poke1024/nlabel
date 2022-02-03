from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(
    name='nlabel',
    version='0.0.1.dev0',
    packages=find_packages(),
    description='nlabel tagging and embeddings library',
    url='https://github.com/poke1024/nlabel',
    license=open('LICENSE', 'r').read(),
    author='Bernhard Liebl',
    author_email='poke1024@gmx.de',
    install_requires=required,
    long_description=open('README.md').read(),
    include_package_data=True,
)
