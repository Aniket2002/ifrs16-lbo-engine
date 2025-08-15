from setuptools import setup, find_packages

setup(
    name="ifrs16-lbo-engine",
    version="1.0.0",
    description="Fast analytic LBO covenant optimization with IFRS-16 dual conventions",
    author="Academic Research Team",
    author_email="research@university.edu",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "jupyter>=1.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "ifrs16-case-study=analysis.scripts.case_study_accor:run_accor_case_study",
            "ifrs16-evaluation=analysis.scripts.evaluation_protocol:run_evaluation_protocol"
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics"
    ]
)
