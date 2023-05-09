# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2023-05-09

### Fixed

- Incorrect attributes `Y.shape` and `Y.size` for a multicomplex or multidual
  array `Y` generated though the reciprocal operation `Y = 1. / X`, where `X`
  is also a multicomplex or multidual array. The reciprocal operation is
  calculated by the method `marray.__rtruediv__` in the multicomplex module,
  `mcomplex.py`, and the multidual module, `mdual.py`.
