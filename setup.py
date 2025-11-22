from pathlib import Path
import re
from setuptools import setup, find_packages

ROOT = Path(__file__).parent


def read_version():
    init_text = (ROOT / "pysml" / "__init__.py").read_text()
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", init_text)
    if not match:
        raise RuntimeError("Unable to find version string in pysml/__init__.py")
    return match.group(1)


setup(
    name="pysml",
    version=read_version(),
    description="Python Strategic Hardware-Independent Learning",
    long_description=(ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy>=1.20"],
)
