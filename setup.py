from setuptools import setup, find_packages

VERSION = "1.2"

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
    entry_points={'console_scripts': ['deepImpute=deepimpute.deepImpute:deepImpute']},
    license="MIT",
    packages=find_packages(exclude=["examples", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "numpy",
        "pandas>=1.0",
        "scipy",
        "sklearn",
        "tensorflow>=2.0",
        "configparser",
        "keras"
    ],
)
