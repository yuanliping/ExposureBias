from setuptools import setup, find_packages, Extension

bleu = Extension(
    'CTG.libbleu',
    sources=[
        'CTG/clib/libbleu/libbleu.cpp',
        'CTG/clib/libbleu/module.cpp',
    ],
    extra_compile_args=['-std=c++11'],
)

with open('README.md') as f:
    readme = f.read()

with open('requirements.txt') as f:
    reqs = f.read()

setup(
    name='CTG',
    version='0.1.0',
    description='Controllable Text Generation (implemented by Jiangtao Feng)',
    long_description=readme,
    #install_requires=reqs.strip().split('\n'),
    packages=find_packages(),
    ext_modules=[bleu],
)
