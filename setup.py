import os
from setuptools import setup


def readlines(nm):
    with open(nm) as f:
        contents = f.read()
    return [l for l in contents.split('\n') if len(l) > 0]


if __name__ == "__main__":
    root = os.path.realpath(os.path.dirname(__file__))
    loadname = os.path.join(root, "requirements.txt")
    install_requires = readlines(loadname)

    setup(
        name='bayesregress',
        version='0.0.1',
        packages=[
            'bayesregress',
        ],
        install_requires=install_requires,
    )
