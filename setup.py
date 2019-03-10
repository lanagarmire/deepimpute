from setuptools import setup, find_packages

VERSION = "0.0.1"

setup(
    name="deepimpute",
    version=VERSION,
    description="scRNA-seq imputation",
    long_description=""" Deep learning (Tensorflow) based method for single cell RNA-seq data imputation. """,
    classifiers=[],
    keywords="Neural network, Deep Learning, scRNA-seq, single-cell, imputation",
    author="Cedric Aridakessian",
    author_email="carisdak@hawaii.edu",
    url="",
    license="MIT",
    packages=find_packages(exclude=["examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "tensorflow",
        "configparser",
        "keras"
    ],
)
