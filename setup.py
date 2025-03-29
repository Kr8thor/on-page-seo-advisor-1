from setuptools import setup, find_packages

setup(
    name="seo_analyzer",
    version="0.1.0",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "httpx",
        "parsel",
        "readability-lxml",
        "textstat",
        "lxml_html_clean"
    ]
) 