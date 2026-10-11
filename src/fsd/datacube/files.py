# Two different artifacts share one file name, "metadata.pickle.npy": a datacube's metadata
# (written by the builder next to `datacube.npy`) and a flattened training set's metadata
# (written by `flatten` next to `data.npy`). Each site names the one it means.
CUBE_NAME = "datacube.npy"
CUBE_METADATA_NAME = "metadata.pickle.npy"
TRAINING_METADATA_NAME = "metadata.pickle.npy"
