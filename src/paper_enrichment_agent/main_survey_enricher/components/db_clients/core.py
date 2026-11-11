"""Contains the storage-agnostic contract of the database clients and its implementations.

A :class:`DBClient` is an asynchronous object store addressed by POSIX-like paths (e.g.
`documents/<paper_id>/metadata.json`). The domain-specific clients, such as
:class:`~paper_enrichment_agent.main_survey_enricher.components.db_clients.doc_db_client.DocDBClient`,
depend only on this contract, so that the storage backend can be swapped or mocked.

Exported classes:
    DBClientException
    DBClient
    S3DBClient
"""

import abc
import asyncio
import contextlib
import pathlib
from collections.abc import AsyncGenerator
from typing import BinaryIO, Protocol, Self

import pydantic
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError as BotocoreClientError
from mypy_boto3_s3 import S3Client


class DBClientException(Exception):
    """Raised by the :class:`DBClient` implementations when a database operation fails."""


class DBClient(Protocol):
    """Contract of an asynchronous object store addressed by POSIX-like paths.

    The implementations provide the primitive operations (reading, writing, listing and deleting
    objects), while the contract provides helpers built on top of them for storing pydantic
    models. Implementations may either subclass the contract explicitly, which makes them inherit
    the helpers, or implement the whole interface structurally.

    Every operation raises :class:`DBClientException` on failure. The operations are asynchronous
    and must not block the event loop.
    """

    @abc.abstractmethod
    async def object_exists(self, path: str) -> bool:
        """Tells whether an object is present at the given path.

        Args:
            path: The path of the object.

        Returns:
            True if the object is present, False otherwise.
        """

    @abc.abstractmethod
    async def read_object(self, path: str) -> bytes:
        """Reads the whole content of the object at the given path.

        Args:
            path: The path of the object.

        Returns:
            The content of the object.

        Raises:
            DBClientException: If the object is missing or cannot be read.
        """

    @abc.abstractmethod
    async def read_to_stream(self, path: str, stream: BinaryIO) -> None:
        """Writes the content of the object at the given path to the stream.

        The content is transferred in chunks, without loading it whole into memory.

        Args:
            path: The path of the object.
            stream: The writable binary stream receiving the content.

        Raises:
            DBClientException: If the object is missing or cannot be read.
        """

    @abc.abstractmethod
    async def write_object(self, path: str, data: bytes, content_type: str) -> None:
        """Writes the data to the object at the given path, replacing any existing content.

        Args:
            path: The path of the object.
            data: The content to be written.
            content_type: The MIME type of the content.
        """

    @abc.abstractmethod
    async def write_stream(self, path: str, stream: BinaryIO, content_type: str) -> None:
        """Writes the content read from the stream to the object at the given path.

        The stream is read until exhausted, without loading its whole content into memory.

        Args:
            path: The path of the object.
            stream: The readable binary stream providing the content.
            content_type: The MIME type of the content.
        """

    @abc.abstractmethod
    async def delete_object(self, path: str) -> None:
        """Deletes the object at the given path.

        Args:
            path: The path of the object.
        """

    @abc.abstractmethod
    async def list_objects(self, prefix: str) -> list[str]:
        """Lists the paths of all objects whose paths start with the given prefix.

        The listing is recursive, i.e. it includes objects nested arbitrarily deep.

        Args:
            prefix: The prefix of the listed paths, e.g. `documents/<paper_id>/images/`.

        Returns:
            The full paths of the matching objects.
        """

    @abc.abstractmethod
    async def list_subdirectories(self, prefix: str) -> list[str]:
        """Lists the direct subdirectories of the directory given by the prefix.

        Args:
            prefix: The path of the directory, ending with `/`, e.g. `surveys/`.

        Returns:
            The full paths of the subdirectories, ending with `/`, e.g. `['surveys/abc/']`.
        """

    async def upload_file(self, local_path: pathlib.Path, path: str, content_type: str) -> None:
        """Streams the content of a local file to the object at the given path.

        Args:
            local_path: The path of the local file to be uploaded.
            path: The path of the object.
            content_type: The MIME type of the content.

        Raises:
            DBClientException: If the local file cannot be opened or the upload fails.
        """

        try:
            with local_path.open('rb') as stream:
                await self.write_stream(path=path, stream=stream, content_type=content_type)

        except OSError as e:
            raise DBClientException(f'Failed to read the local file {local_path}: {e}') from e

    async def download_to_file(self, path: str, local_path: pathlib.Path) -> None:
        """Streams the content of the object at the given path to a local file.

        The local file is created (or truncated) on the host running the client. A partially
        written file is removed if the download fails.

        Args:
            path: The path of the object.
            local_path: The path of the local file to be written. Its parent directory must exist.

        Raises:
            DBClientException: If the local file cannot be written or the download fails.
        """

        try:
            with local_path.open('wb') as stream:
                await self.read_to_stream(path=path, stream=stream)

        except OSError as e:
            local_path.unlink(missing_ok=True)
            raise DBClientException(f'Failed to write the local file {local_path}: {e}') from e

        except DBClientException:
            local_path.unlink(missing_ok=True)
            raise

    async def upload_model(self, model: pydantic.BaseModel, path: str) -> None:
        """Serializes the pydantic model to JSON and stores it at the given path.

        Args:
            model: The model to be stored.
            path: The path of the object.
        """

        await self.write_object(
            path=path, data=model.model_dump_json().encode('utf-8'), content_type='application/json'
        )

    async def get_model[T: pydantic.BaseModel](self, path: str, model_type: type[T]) -> T:
        """Reads the JSON object at the given path and validates it as a pydantic model.

        Args:
            path: The path of the object.
            model_type: The type of the model to be validated.

        Returns:
            The validated model.

        Raises:
            DBClientException: If the object cannot be read or is not a valid model.
        """

        json_bytes = await self.read_object(path=path)

        try:
            return model_type.model_validate_json(json_bytes)

        except pydantic.ValidationError as e:
            raise DBClientException(f'Failed to decode JSON from database at {path}: {e}') from e

    @contextlib.asynccontextmanager
    async def model_reference[T: pydantic.BaseModel](
        self, path: str, model_type: type[T]
    ) -> AsyncGenerator[T, None]:
        """Returns an async context manager yielding a model synchronized with the database.

        The model is read on entering the context and written back on a successful exit, so the
        modifications made within the context are persisted. The modifications are discarded if
        the context exits with an exception.

        Args:
            path: The path of the object.
            model_type: The type of the model to be validated.
        """

        model = await self.get_model(path=path, model_type=model_type)

        yield model

        await self.upload_model(model=model, path=path)


class S3DBClient(DBClient):
    """Implements :class:`DBClient` on top of a single bucket of an S3-compatible storage.

    The client should be created with `create`, which creates the bucket if it is not present yet,
    so that the operations can assume that the bucket exists. The paths are used as the object
    keys. Since the underlying boto3 client is blocking, its calls are run in separate threads.
    """

    # Error codes returned by S3 for missing buckets / objects.
    _MISSING_BUCKET_CODES = ('404', 'NoSuchBucket')
    _MISSING_OBJECT_CODES = ('404', 'NoSuchKey')

    def __init__(self, s3_client: S3Client, bucket_name: str) -> None:

        self._s3_client = s3_client
        self._bucket = bucket_name

    @classmethod
    def create(cls, s3_client: S3Client, bucket_name: str) -> Self:
        """Creates the client and ensures that its bucket exists.

        The creation is blocking, as it is meant to happen once on the service's startup.

        Args:
            s3_client: The client of the S3-compatible storage.
            bucket_name: The name of the bucket storing the objects.

        Returns:
            The client ready to perform the operations.

        Raises:
            DBClientException: If checking the presence of the bucket or creating it fails.
        """

        try:
            s3_client.head_bucket(Bucket=bucket_name)

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') not in cls._MISSING_BUCKET_CODES:
                raise DBClientException(
                    f'Failed to check the presence of the bucket {bucket_name}: {e}'
                ) from e

            try:
                s3_client.create_bucket(Bucket=bucket_name)

            except BotocoreClientError as e:
                raise DBClientException(f'Failed to create the bucket {bucket_name}: {e}') from e

        return cls(s3_client=s3_client, bucket_name=bucket_name)

    async def object_exists(self, path: str) -> bool:
        """See :meth:`DBClient.object_exists`."""

        try:
            await asyncio.to_thread(self._s3_client.head_object, Bucket=self._bucket, Key=path)

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') in self._MISSING_OBJECT_CODES:
                return False

            raise DBClientException(
                f'Failed to check the presence of the object at {path}: {e}'
            ) from e

        return True

    async def read_object(self, path: str) -> bytes:
        """See :meth:`DBClient.read_object`."""

        try:
            return await asyncio.to_thread(self._read_object_blocking, path=path)

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to read the object at {path}: {e}') from e

    async def read_to_stream(self, path: str, stream: BinaryIO) -> None:
        """See :meth:`DBClient.read_to_stream`."""

        try:
            await asyncio.to_thread(
                self._s3_client.download_fileobj, Bucket=self._bucket, Key=path, Fileobj=stream
            )

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to read the object at {path} to stream: {e}') from e

    async def write_object(self, path: str, data: bytes, content_type: str) -> None:
        """See :meth:`DBClient.write_object`."""

        try:
            await asyncio.to_thread(
                self._s3_client.put_object,
                Bucket=self._bucket,
                Key=path,
                Body=data,
                ContentType=content_type,
            )

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to write the object at {path}: {e}') from e

    async def write_stream(self, path: str, stream: BinaryIO, content_type: str) -> None:
        """See :meth:`DBClient.write_stream`."""

        try:
            await asyncio.to_thread(
                self._s3_client.upload_fileobj,
                stream,
                Bucket=self._bucket,
                Key=path,
                ExtraArgs={'ContentType': content_type},
            )

        except (BotocoreClientError, S3UploadFailedError) as e:
            raise DBClientException(f'Failed to write the stream to {path}: {e}') from e

    async def delete_object(self, path: str) -> None:
        """See :meth:`DBClient.delete_object`."""

        try:
            await asyncio.to_thread(self._s3_client.delete_object, Bucket=self._bucket, Key=path)

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to delete the object at {path}: {e}') from e

    async def list_objects(self, prefix: str) -> list[str]:
        """See :meth:`DBClient.list_objects`."""

        try:
            return await asyncio.to_thread(self._list_objects_blocking, prefix=prefix)

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to list the objects under {prefix}: {e}') from e

    async def list_subdirectories(self, prefix: str) -> list[str]:
        """See :meth:`DBClient.list_subdirectories`."""

        try:
            return await asyncio.to_thread(self._list_subdirectories_blocking, prefix=prefix)

        except BotocoreClientError as e:
            raise DBClientException(f'Failed to list the subdirectories of {prefix}: {e}') from e

    def _read_object_blocking(self, path: str) -> bytes:
        """Reads the whole content of the object at the specified path (blocking)."""

        return self._s3_client.get_object(Bucket=self._bucket, Key=path)['Body'].read()

    def _list_objects_blocking(self, prefix: str) -> list[str]:
        """Lists the keys under the prefix, following the pagination (blocking)."""

        paginator = self._s3_client.get_paginator('list_objects_v2')

        return [
            obj['Key']
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix)
            # S3 omits `Contents` from the page when no object matches the prefix.
            for obj in page.get('Contents', [])
        ]

    def _list_subdirectories_blocking(self, prefix: str) -> list[str]:
        """Lists the common prefixes under the prefix, following the pagination (blocking)."""

        paginator = self._s3_client.get_paginator('list_objects_v2')

        return [
            common_prefix['Prefix']
            for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix, Delimiter='/')
            # S3 omits `CommonPrefixes` from the page when there is no subdirectory.
            for common_prefix in page.get('CommonPrefixes', [])
        ]
