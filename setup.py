from setuptools import find_packages, setup

setup(
    name="ribflox",
    packages=["ribflox"],
    version="0.1.0",
    url="https://github.com/noegroup/rigid-flows.git",
    author="Jonas Köhler",
    author_email="jonas.koehler.ks@gmail.com",
    description="flows for rigid bodies",
    python_requires=">3.10",
    # install_requires=["flox @ git+ssh://git@github.com/noegroup/flox.git"],
)
