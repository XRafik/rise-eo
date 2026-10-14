"""Generic registry mechanics (`_core`), used by both `fsd.model.registry` and
`fsd.image.registry`.

`_files.py` (the staged JSON write, the best-effort discard, the content digest) is also
used by `fsd.model.registry`, and its content digest by `fsd.image.digest`.

Spec: specs/56-image-definitions-and-registry.md, §7 Q1.
"""
