from setuptools import setup, find_packages

setup(
    name="mopaqdaps",
    version="0.1.0",
    description="Pitch Shifting Evaluation Framework",
    author="João Pedro Torres, Gabriel Barbosa da Fonseca",
    author_email="joao.silva@sga.pucminas.br",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.5",
        "pandas>=1.1.5",
        "matplotlib>=3.2.2",
        "librosa>=0.9.2",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.5.4",
        "torch>=1.9.0",
        "torchaudio>=0.9.0"
    ],
    entry_points={
        'console_scripts': [
            'mopaqdaps=mopaqdaps.main:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
