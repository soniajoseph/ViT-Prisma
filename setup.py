from setuptools import setup, find_packages


install_requires = [
    'pytest>=6.0',  # Test dependency, but included here to auto-install
    'torch',        # Test dependency
    'numpy',        # Test dependency
    'jaxtyping',    # Test dependency
    'einops',       # Test dependency
    'fancy_einsum', # Test dependency
    'plotly==5.19.0',
    'timm',         # Test dependency
    'transformers', # Test dependency
    'scikit-learn', # Test dependency
]

setup(
    name='vit-prisma',
    version='0.1.3',
    author='Sonia Joseph',
    author_email='soniamollyjoseph@gmail.com',
    description='A Vision Transformer library for mechanistic interpretability.',
    long_description=open('docs/README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/soniajoseph/vit-prisma',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    package_data={
    'vit_prisma': ['visualization/*.html', 'visualization/*.js'],
    # Add other patterns here as needed
},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
    install_requires=install_requires,
    keywords='vision-transformer, clip, multimodal, machine-learning, mechanistic interpretability',
    zip_safe=False,
)
