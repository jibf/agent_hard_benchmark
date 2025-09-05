from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="DrafterBench",
    version="0.1.0",
    description="A benchmark evaluates LLMs' performance in automating drawing revision tasks.",
    author="Yinsheng Li, Zhen Dong, Yi Shao",
    author_email="yinsheng.li@mail.mcgill.ca",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=requirements,
)
