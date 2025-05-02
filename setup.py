from setuptools import setup, find_packages

setup(
    name='AugmentedSocialScientistFork',
    version='2.2.1.post1',
    description='Modified version of AugmentedSocialScientist for custom research purposes',
    author='Antoine Lemor',
    author_email='antoine.lemor@umontreal.ca',
    url='https://github.com/antoinelemor/AugmentedSocialScientist',  # <-- Change to your GitHub URL if you want
    packages=find_packages(where='.'),
    install_requires=[
        'torch>=1.13',
        'transformers>=4.30',
        'pandas>=1.5',
    ],
    python_requires='>=3.8',
)
