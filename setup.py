from setuptools import setup

setup(
    name="m-vllm",
    version="0.1.0",
    description="m-vllm: A vLLM implementation",
    author="",
    author_email="",
    packages=["m_vllm", "m_vllm.kernels", "m_vllm.layers", "m_vllm.data_classes", "m_vllm.models"],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "pybind11",
    ],
    package_dir={"m_vllm": "m-vllm"},
)
