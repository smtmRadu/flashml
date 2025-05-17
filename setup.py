from setuptools import setup, find_packages
from flashml import __version__

setup(
    name="flashml",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "torch",
        "torchtune",
        "torchao",
        "torchinfo",
        "tqdm",
        "psutil",
        "windows-curses",
        "playsound==1.3.0",
        "pillow",
        "nltk",
        "pynvml",
        "scikit-learn",
        "pynput",
        "pyautogui",
        "pyperclip",
        "tkinterdnd2",
        "joblib",
        "polars",
        "pandas",
        # not dependence but nice to have installed
        "python-dotenv",
        "ollama",
        "torch-optimi",
        "gymnasium",
        "transformers",
    ],
    author="kbradu",
)
