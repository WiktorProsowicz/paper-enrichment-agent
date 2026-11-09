"""Contains the environment shared by the document serializers.

A :class:`SerializationEnvironment` is the prepared working area of a single serialization of a
document. The caller (e.g. the `main_survey_enricher` service) creates it within a directory it
owns, places the document's images at the paths given by the environment, and passes the
environment to a serializer, which writes its output into the environment's `output_dir` and packs
it into the archive at `archive_path`.

Exported classes:
    SerializationEnvironment
"""

import pathlib
from typing import Self

from paper_enrichment_agent.common.document_manipulators import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models


class SerializationEnvironment:
    """Wraps a document together with the local directory it is serialized within.

    The environment should be created with `create`, which prepares the directory structure, so
    that the caller can directly write the document's images to the paths returned by
    `image_path_for_component`. The environment does not own the working directory, so the caller
    is responsible for removing it once the serialization's results are no longer needed. A single
    environment should be used for a single serialization.

    The creation raises `SerializationEnvironmentError` on failure.
    """

    class SerializationEnvironmentError(Exception):
        """Base class for exceptions raised by the `SerializationEnvironment`."""

    # Subdirectory of the working directory holding the files packed into the archive.
    _OUTPUT_DIR_NAME = 'content'
    # Subdirectory of the output directory holding the document's images.
    _IMAGES_DIR_NAME = 'images'
    # Name of the archive created within the working directory.
    _ARCHIVE_NAME = 'document.zip'

    def __init__(self, document: doc_models.Document, work_dir: pathlib.Path) -> None:

        self._document = document
        self._work_dir = work_dir

        # Maps the paths of the image subfigures to the names of their local files.
        self._image_file_names = {
            path: f'{path.strip("/").replace("/", "-")}.{component.image_src.split(".")[-1]}'
            for path, component in DocumentGetter(document).iter_components_with_paths()
            if isinstance(component, doc_models.ImgSubfigure)
        }

    @classmethod
    def create(cls, document: doc_models.Document, work_dir: pathlib.Path) -> Self:
        """Creates the environment and prepares its directory structure.

        The creation is blocking, but it only creates a few directories.

        Args:
            document: The document to be serialized.
            work_dir: The directory to serialize the document within. It should be empty and
                dedicated to a single serialization. It is created if it does not exist.

        Returns:
            The environment ready for placing the images and running a serializer.

        Raises:
            SerializationEnvironmentError: If the directory structure cannot be created.
        """

        environment = cls(document=document, work_dir=work_dir)

        try:
            (environment.output_dir / cls._IMAGES_DIR_NAME).mkdir(parents=True, exist_ok=True)

        except OSError as e:
            raise cls.SerializationEnvironmentError(
                f'Failed to prepare the serialization environment in {work_dir}: {e}'
            ) from e

        return environment

    @property
    def document(self) -> doc_models.Document:
        """Returns the document to be serialized."""

        return self._document

    @property
    def output_dir(self) -> pathlib.Path:
        """Returns the directory the serializer puts its results into.

        The whole content of the directory, including the images, is packed into the archive.
        """

        return self._work_dir / self._OUTPUT_DIR_NAME

    @property
    def archive_path(self) -> pathlib.Path:
        """Returns the path of the archive the serializer packs its results into."""

        return self._work_dir / self._ARCHIVE_NAME

    @property
    def image_component_paths(self) -> list[str]:
        """Returns the paths of all image subfigures of the document, in the document order."""

        return list(self._image_file_names)

    def image_path_for_component(self, component_path: str) -> pathlib.Path:
        """Returns the local path of the image of the given image subfigure.

        The path lies within the `images` subdirectory of `output_dir`. The method does not
        check whether the image file exists, since the images are placed by the caller.

        Args:
            component_path: The path of an image subfigure in the document.

        Returns:
            The local path of the image file.

        Raises:
            ValueError: If no component exists at the given path or if the component is not an
                image subfigure.
        """

        component = DocumentGetter(self._document).get_component_by_path(component_path)

        if not isinstance(component, doc_models.ImgSubfigure):
            raise ValueError(f'Component at path {component_path} is not an image subfigure.')

        return self.output_dir / self._IMAGES_DIR_NAME / self._image_file_names[component_path]
