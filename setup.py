import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

print(setuptools.find_packages())

setuptools.setup(
    name="fss",
    version="0.0.1",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'torch', 'numpy', 'scipy', 'nibabel', 'imageio', 'torchvision', 'tqdm', 'Pillow',
        'autorepr', 'scikit-learn', 'pydicom', 'joblib', 'pyyaml', 'easydict', 'SimpleITK'
    ],
)
