from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='nlabel',
    version='0.0.1a',
    description='nlabel tagging and embeddings library',
    author='Bernhard Liebl',
    author_email='poke1024@gmx.de',
    install_requires=required,
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
)
