from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="healthai",
    version="0.1.0",
    author="Bozheng Long",
    author_email="im.bzlong@gmail.com",
    description="Medical diagnosis system using large language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BozhengLong/HealthAI2025-ClinicalData",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.37.2",
        "peft>=0.7.1",
        "datasets>=2.16.1",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "healthai-train=healthai.cli.train:main",
            "healthai-infer=healthai.cli.infer:main",
        ],
    },
) 