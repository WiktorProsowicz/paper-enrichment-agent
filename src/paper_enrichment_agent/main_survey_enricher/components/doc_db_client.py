"""Contains the client class that communicates with the document database."""

import contextlib
import json
import uuid
from collections.abc import Generator

import pydantic
import requests
from botocore.exceptions import ClientError as BotocoreClientError
from mypy_boto3_s3 import S3Client

from paper_enrichment_agent.common.document_getter import DocumentGetter
from paper_enrichment_agent.common.models import document as doc_models
from paper_enrichment_agent.common.models.misc import DocumentMetadata, SurveyMetadata


class DocDBClient:
    """Handles read/write requests to the document database."""

    class DocDBClientError(Exception):
        """Base class for exceptions raised by the `DocDBClient`."""

    def __init__(self, s3_client: S3Client) -> None:

        self._s3_client = s3_client
        self._s3_bucket = 'document_database'

    def document_exists(self, paper_id: str) -> bool:
        """Tells whether a document with the given id is present in the database.

        Args:
            paper_id: The identifier of the document to be looked up.

        Returns:
            True if the document is present in the database, False otherwise.

        Raises:
            DocDBClientError: If there is an error while checking the presence of the document.
        """

        try:
            self._s3_client.head_object(
                Bucket=self._s3_bucket, Key=f'documents/{paper_id}/metadata.json'
            )

        except BotocoreClientError as e:
            if e.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey'):
                return False

            raise self.DocDBClientError(
                f'Failed to check the presence of the document {paper_id} in database: {e}'
            ) from e

        return True

    def add_document(
        self, document: doc_models.Document, name: str, description: str
    ) -> DocumentMetadata:
        """Adds a document together with its metadata to the document database.

        Args:
            document: The document to be added to the database.
            name: The name of the document.
            description: The description of the document.

        Returns:
            The metadata of the freshly added document.

        Raises:
            DocDBClientError: If there is an error while adding the document to the database.
        """

        doc_metadata = DocumentMetadata(name=name, description=description, images={})

        self._download_and_store_images(document=document, doc_metadata=doc_metadata)

        self._upload_model_to_db(
            doc_metadata, path=f'documents/{doc_metadata.paper_id}/metadata.json'
        )
        self._upload_model_to_db(document, path=f'documents/{doc_metadata.paper_id}/document.json')

        return doc_metadata

    def register_as_survey(self, paper_id: str) -> SurveyMetadata:
        """Marks the document with the given id as a survey.

        The caller is responsible for ensuring that the document is present in the database,
        see `document_exists`.

        Args:
            paper_id: The identifier of an already added document.

        Returns:
            The metadata of the freshly created survey.

        Raises:
            DocDBClientError: If there is an error while saving the survey metadata to the database.
        """

        survey_metadata = SurveyMetadata(paper_id=paper_id, referenced_docs={})

        self._upload_model_to_db(survey_metadata, path=f'surveys/{paper_id}/metadata.json')

        return survey_metadata

    def get_available_surveys(self) -> list[SurveyMetadata]:
        """Returns a list of all available surveys in the document database.

        Returns:
            A list of `SurveyMetadata` objects representing the available surveys.

        Raises:
            DocDBClientError: If there is an error while retrieving the surveys from the database.
        """

        try:
            response = self._s3_client.list_objects_v2(
                Bucket=self._s3_bucket, Prefix='surveys/', Delimiter='/'
            )

            prefixes = response['CommonPrefixes']

        except (BotocoreClientError, KeyError) as e:
            raise self.DocDBClientError(f'Failed to list surveys in database: {e}') from e

        survey_metadata_list = []

        for prefix in prefixes:
            survey_prefix = prefix['Prefix']
            metadata_path = f'{survey_prefix}metadata.json'

            survey_metadata_list.append(
                self._get_model_from_db(path=metadata_path, model_type=SurveyMetadata)
            )

        return survey_metadata_list

    def add_referenced_document(
        self, survey_id: str, reference_id: str, referenced_paper_id: str
    ) -> None:
        """Adds a referenced document to a survey in the database.

        The caller is responsible for ensuring that both the survey and the referenced document
        are present in the database, see `document_exists`.

        Args:
            survey_id: The identifier of the survey in the database.
            reference_id: The identifier of the reference in the survey.
            referenced_paper_id: The identifier of the referenced document in the database.

        Raises:
            DocDBClientError: If there is an error while adding the referenced document to
                the survey in the database.
        """

        survey_metadata_path = f'surveys/{survey_id}/metadata.json'

        with self._get_model_reference(
            path=survey_metadata_path, model_type=SurveyMetadata
        ) as survey:
            survey.referenced_docs[reference_id] = referenced_paper_id

    def delete_survey(self, survey_id: str) -> None:
        """Deletes a survey and its referenced documents from the database.

        The caller is responsible for ensuring that the survey is present in the database,
        see `document_exists`.

        Args:
            survey_id: The identifier of the survey to be deleted.

        Raises:
            DocDBClientError: If there is an error while deleting the survey from the database.
        """

        try:
            metadata = self._get_model_from_db(
                path=f'surveys/{survey_id}/metadata.json', model_type=SurveyMetadata
            )

            for referenced_doc_id in metadata.referenced_docs.values():
                self._delete_document(paper_id=referenced_doc_id)

            self._s3_client.delete_object(
                Bucket=self._s3_bucket, Key=f'surveys/{survey_id}/metadata.json'
            )

            self._delete_document(paper_id=survey_id)

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to delete survey {survey_id} from database: {e}'
            ) from e

    def _delete_document(self, paper_id: str) -> None:
        """Deletes a document and its images from the database."""

        try:
            self._s3_client.delete_object(
                Bucket=self._s3_bucket, Key=f'documents/{paper_id}/metadata.json'
            )

            self._s3_client.delete_object(
                Bucket=self._s3_bucket, Key=f'documents/{paper_id}/document.json'
            )

            list_images_response = self._s3_client.list_objects_v2(
                Bucket=self._s3_bucket, Prefix=f'documents/{paper_id}/images/'
            )
            for obj in list_images_response.get('Contents', []):
                self._s3_client.delete_object(Bucket=self._s3_bucket, Key=obj['Key'])

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to delete document {paper_id} from database: {e}'
            ) from e

    def _download_and_store_images(
        self, document: doc_models.Document, doc_metadata: DocumentMetadata
    ) -> None:
        """Downloads and stores the images of a document in the database."""

        doc_getter = DocumentGetter(document=document)

        for image in doc_getter.iter_components_of_type(doc_models.ImgSubfigure):
            image_extension = image.image_src.split('.')[-1]

            image_db_id = uuid.uuid4().hex
            image_db_path = (
                f'documents/{doc_metadata.paper_id}/images/{image_db_id}.{image_extension}'
            )

            try:
                with requests.get(image.image_src, stream=True) as response:
                    response.raise_for_status()

                    self._s3_client.upload_fileobj(
                        response.raw,
                        Bucket=self._s3_bucket,
                        Key=image_db_path,
                        ExtraArgs={
                            'ContentType': response.headers.get(
                                'Content-Type', 'application/octet-stream'
                            )
                        },
                    )

            except requests.RequestException as e:
                raise self.DocDBClientError(
                    f'Failed to download image from {image.image_src}: {e}'
                ) from e

            except Exception as e:
                raise self.DocDBClientError(
                    f'Failed to upload image to database at {image_db_path}: {e}'
                ) from e

            doc_metadata.images[doc_getter.get_path_of_component(image)] = image_db_path

    @contextlib.contextmanager
    def _get_model_reference[T: pydantic.BaseModel](
        self, path: str, model_type: type[T]
    ) -> Generator[T, None, None]:
        """Returns a context manager that yields a Pydantic model reference from the database.

        The context manager ensures that the reference is properly synchronized with the
        database.
        """

        model = self._get_model_from_db(path=path, model_type=model_type)

        yield model

        self._upload_model_to_db(model=model, path=path)

    def _get_model_from_db[T: pydantic.BaseModel](self, path: str, model_type: type[T]) -> T:
        """Retrieves a Pydantic model from the database at the specified path."""

        try:
            json_file = self._s3_client.get_object(Bucket=self._s3_bucket, Key=path)

        except BotocoreClientError as e:
            raise self.DocDBClientError(
                f'Failed to retrieve JSON from database at {path}: {e}'
            ) from e

        try:
            model = model_type.model_validate_json(json_file['Body'].read().decode('utf-8'))

        except pydantic.ValidationError as e:
            raise self.DocDBClientError(
                f'Failed to decode JSON from database at {path}: {e}'
            ) from e

        return model

    def _upload_model_to_db(self, model: pydantic.BaseModel, path: str) -> None:
        """Uploads a Pydantic model to the database at the specified path."""

        try:
            self._s3_client.put_object(
                Bucket=self._s3_bucket,
                Key=path,
                Body=json.dumps(model.model_dump()).encode('utf-8'),
            )

        except BotocoreClientError as e:
            raise self.DocDBClientError(f'Failed to upload model to database at {path}: {e}') from e
