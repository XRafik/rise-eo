"""Satellite data sources (`cdse`, `mpc`). See specs/01-sources.md.

The source contract is a documented function signature (duck typing), not an abstract
base class: a "source" is any module exposing a `download(...)` shaped like `cdse.download`.
"""
