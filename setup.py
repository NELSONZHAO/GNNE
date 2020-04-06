import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gnne",
    version="0.0.1",
    author="nelsonzhao",
    author_email="dutzhaoyeyu@163.com",
    description="A Graph Neural Network Embedding Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NELSONZHAO/GNNE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)