from setuptools import setup, find_packages

setup(
    name='augmented-social-scientist',
    version='2.2.1-custom',
    description='Modified version of AugmentedSocialScientist for custom research purposes',
    author='Antoine Lemor',
    author_email='antoine.lemor@umontreal.ca',
    url='https://github.com/ton_username/AugmentedSocialScientist',  # à changer
    packages=find_packages(),
    install_requires=[
        'torch>=1.13',
        'transformers>=4.30',
        'pandas>=1.5',
    ],
    python_requires='>=3.8',
)