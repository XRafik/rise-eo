"""Generic registry mechanics (`_core`), used by both `rise_eo.model.registry` and
`rise_eo.image.registry`.

`_files.py` (the staged JSON write, the best-effort discard, the content digest) is also
used by `rise_eo.model.registry`, and its content digest by `rise_eo.image.digest`.

Spec: specs/56-image-definitions-and-registry.md, §7 Q1.
"""
