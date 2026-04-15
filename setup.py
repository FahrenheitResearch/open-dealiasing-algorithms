from setuptools import setup
from setuptools_rust import Binding, RustExtension


setup(
    rust_extensions=[
        RustExtension(
            "open_dealias._rust",
            path="rust/open_dealias_py/Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
        )
    ],
    zip_safe=False,
)
