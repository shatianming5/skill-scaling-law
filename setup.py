from setuptools import setup, find_packages

setup(
    name="skill-scaling-law",
    version="0.1.0",
    description="Skill Scaling Laws for AI Agents",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "pyyaml>=6.0",
        "anthropic>=0.39",
        "openai>=1.50",
    ],
)
