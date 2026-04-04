"""
setup.py – CAD Automation Pipeline
Editable install: pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="cad_automation",
    version="0.1.0",
    description=(
        "Hybrid CAD pipeline: STEP → Knowledge Graph + GNN → ISO-compliant DXF"
    ),
    author="M.Tech Research",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "data*", "checkpoints*", "logs*"]),
    install_requires=[
        "numpy<2.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0.1",
        "loguru>=0.7.2",
        "tqdm>=4.66.1",
        "click>=8.1.7",
        "h5py>=3.10.0",
        "rich>=13.7.0",
        "ezdxf>=1.1.4",
        "python-constraint>=1.4.0",
        "neo4j>=5.14.1",
    ],
    entry_points={
        "console_scripts": [
            "cad-run=pipelines.run_all:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
    ],
)
