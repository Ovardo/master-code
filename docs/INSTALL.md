## GTSAM installation

The factor graph backend uses GTSAM through its Python bindings. The stable and development package variants can be installed with:

```bash
pip install gtsam
pip install gtsam-develop
```

The covariance-recovery experiments require functionality from the GTSAM development version.

During early development, GTSAM was built from source because some required C++ covariance-recovery functions were not exposed through the Python bindings. These functions have since been added to the official development version, so the local modifications are no longer required.

Pre-built wheel availability depends on the operating system and architecture. If no compatible wheel is available, GTSAM can be built from source. Windows users may alternatively use WSL or a prepared container environment.

### Building GTSAM from source

Building from source is useful when functionality in the underlying C++ library must be modified or exposed through the Python wrapper. Release-mode builds should be used for runtime measurements because debug builds contain additional checks and may be substantially slower.

The Python wrapper is generated from GTSAM's C++ interface files. It can also be built with Doxygen-derived docstrings to improve the documentation available through the Python interface.

Further instructions are available in the official GTSAM documentation:

- [Installation](https://github.com/borglab/gtsam/blob/develop/INSTALL.md)
- [Python wrapper](https://github.com/borglab/gtsam/blob/develop/python/README.md)
- [Development guide](https://github.com/borglab/gtsam/blob/develop/DEVELOP.md)