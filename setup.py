from setuptools import setup, find_packages

setup(
    name="ifrs16-lbo-engine",
    version="1.0.0",
    description="IFRS-16 covenant benchmarking with analytic screening and full simulation",
    author="Aniket Bhardwaj",
    author_email="bhardwaj.aniket2002@gmail.com",
    packages=find_packages(where="src") + find_packages(include=["analysis", "analysis.*"]),
    package_dir={"": "src", "analysis": "analysis"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "numpy-financial>=1.0.0",
    ],
    extras_require={"dev": ["pytest>=6.0.0", "jupyter>=1.0.0"]},
    entry_points={
        "console_scripts": [
            "ifrs16-case-study=analysis.scripts.case_study_accor:run_accor_case_study",
            "ifrs16-benchmark=analysis.run_benchmark:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
