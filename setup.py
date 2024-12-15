from setuptools import setup, find_packages

setup(
    name="fadtk-versa",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "encodec>=0.1.1",
        "laion-clap>=1.1.6",
        "nnaudio>=0.3.2",
        "hypy_utils",
        "msclap @ git+https://github.com/ftshijt/CLAP.git",
        "resampy"
    ],
    author="Jiatong Shi (Adapated from Microsoft/fadtk)",
    author_email="ftshijt@gmail.com",
    description="Adapated package for fadtk for versa toolkit",
    url="https://github.com/ftshijt/fadtk.git",
    keywords="audio and music evaluation",
)
