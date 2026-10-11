"""The artifact file-name constants equal the literals every consumer and run-book reads."""

from fsd.datacube import files


def test_artifact_names_equal_the_literals():
    assert files.CUBE_NAME == "datacube.npy"
    assert files.CUBE_METADATA_NAME == "metadata.pickle.npy"
    assert files.TRAINING_METADATA_NAME == "metadata.pickle.npy"
