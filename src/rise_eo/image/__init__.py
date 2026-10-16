"""`rise_eo.image` — declarative AML node-image definitions and a registry for them.

An `ImageDefinition` is data, not a Dockerfile: what base image, what fsd reference, what
extras. It renders a Dockerfile and a build context; it never builds one and never touches
the network. Building is `rise_eo.aml`'s job -- backend-specific, and the only module in this
area that is.
"""

from rise_eo.image.definition import ImageDefinition

__all__ = ["ImageDefinition"]
