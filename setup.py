import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ADMMFTIRE", # Replace with your own username
    version="0.0.2",
    author="Jiaying Weng",
    author_email="jweng@bentley.edu",
    description="An ADMM-FT-IRE package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/338gaga/ADMM-FT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

