import io

from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError as BotocoreClientError

import pytest
from unittest.mock import MagicMock

from paper_enrichment_agent.main_survey_enricher.components.db_clients.core import (
    DBClientException,
    S3DBClient,
)


def _client_error(code: str, operation: str) -> BotocoreClientError:
    return BotocoreClientError({'Error': {'Code': code}}, operation)


class TestS3DBClientCreate:
    def test_skips_creation_for_present_bucket(self):

        mock_s3_client = MagicMock()

        S3DBClient.create(s3_client=mock_s3_client, bucket_name='document-database')

        mock_s3_client.head_bucket.assert_called_once_with(Bucket='document-database')
        mock_s3_client.create_bucket.assert_not_called()

    @pytest.mark.parametrize('error_code', ['404', 'NoSuchBucket'])
    def test_creates_missing_bucket(self, error_code):

        mock_s3_client = MagicMock()
        mock_s3_client.head_bucket.side_effect = _client_error(error_code, 'HeadBucket')

        S3DBClient.create(s3_client=mock_s3_client, bucket_name='document-database')

        mock_s3_client.create_bucket.assert_called_once_with(Bucket='document-database')

    def test_raises_when_presence_check_fails(self):

        mock_s3_client = MagicMock()
        mock_s3_client.head_bucket.side_effect = _client_error('AccessDenied', 'HeadBucket')

        with pytest.raises(DBClientException, match='Failed to check the presence of the bucket'):
            S3DBClient.create(s3_client=mock_s3_client, bucket_name='document-database')

        mock_s3_client.create_bucket.assert_not_called()

    def test_raises_when_creation_fails(self):

        mock_s3_client = MagicMock()
        mock_s3_client.head_bucket.side_effect = _client_error('404', 'HeadBucket')
        mock_s3_client.create_bucket.side_effect = _client_error('AccessDenied', 'CreateBucket')

        with pytest.raises(DBClientException, match='Failed to create the bucket'):
            S3DBClient.create(s3_client=mock_s3_client, bucket_name='document-database')


class TestS3DBClient:
    @pytest.fixture
    def mock_s3_client(self) -> MagicMock:

        return MagicMock()

    @pytest.fixture
    def s3_db_client(self, mock_s3_client) -> S3DBClient:

        return S3DBClient(s3_client=mock_s3_client, bucket_name='bucket')

    @pytest.mark.asyncio
    async def test_object_exists_returns_true_for_present_object(
        self, s3_db_client, mock_s3_client
    ):

        assert await s3_db_client.object_exists('a/b.json') is True

        mock_s3_client.head_object.assert_called_once_with(Bucket='bucket', Key='a/b.json')

    @pytest.mark.parametrize('error_code', ['404', 'NoSuchKey'])
    @pytest.mark.asyncio
    async def test_object_exists_returns_false_for_missing_object(
        self, s3_db_client, mock_s3_client, error_code
    ):

        mock_s3_client.head_object.side_effect = _client_error(error_code, 'HeadObject')

        assert await s3_db_client.object_exists('a/b.json') is False

    @pytest.mark.asyncio
    async def test_object_exists_raises_when_presence_check_fails(
        self, s3_db_client, mock_s3_client
    ):

        mock_s3_client.head_object.side_effect = _client_error('AccessDenied', 'HeadObject')

        with pytest.raises(DBClientException, match='Failed to check the presence of the object'):
            await s3_db_client.object_exists('a/b.json')

    @pytest.mark.asyncio
    async def test_read_object_returns_content(self, s3_db_client, mock_s3_client):

        mock_s3_client.get_object.return_value = {'Body': io.BytesIO(b'content')}

        assert await s3_db_client.read_object('a/b.json') == b'content'

        mock_s3_client.get_object.assert_called_once_with(Bucket='bucket', Key='a/b.json')

    @pytest.mark.asyncio
    async def test_read_object_raises_on_failure(self, s3_db_client, mock_s3_client):

        mock_s3_client.get_object.side_effect = _client_error('NoSuchKey', 'GetObject')

        with pytest.raises(DBClientException, match='Failed to read the object at a/b.json'):
            await s3_db_client.read_object('a/b.json')

    @pytest.mark.asyncio
    async def test_write_object_puts_object(self, s3_db_client, mock_s3_client):

        await s3_db_client.write_object('a/b.json', data=b'{}', content_type='application/json')

        mock_s3_client.put_object.assert_called_once_with(
            Bucket='bucket', Key='a/b.json', Body=b'{}', ContentType='application/json'
        )

    @pytest.mark.asyncio
    async def test_write_object_raises_on_failure(self, s3_db_client, mock_s3_client):

        mock_s3_client.put_object.side_effect = _client_error('AccessDenied', 'PutObject')

        with pytest.raises(DBClientException, match='Failed to write the object at a/b.json'):
            await s3_db_client.write_object('a/b.json', data=b'{}', content_type='text/plain')

    @pytest.mark.asyncio
    async def test_write_stream_uploads_fileobj(self, s3_db_client, mock_s3_client):

        stream = io.BytesIO(b'image')

        await s3_db_client.write_stream('a/img.png', stream=stream, content_type='image/png')

        mock_s3_client.upload_fileobj.assert_called_once_with(
            stream, Bucket='bucket', Key='a/img.png', ExtraArgs={'ContentType': 'image/png'}
        )

    @pytest.mark.parametrize(
        'error',
        [_client_error('AccessDenied', 'PutObject'), S3UploadFailedError('Upload failed')],
    )
    @pytest.mark.asyncio
    async def test_write_stream_raises_on_failure(self, s3_db_client, mock_s3_client, error):

        mock_s3_client.upload_fileobj.side_effect = error

        with pytest.raises(DBClientException, match='Failed to write the stream to a/img.png'):
            await s3_db_client.write_stream(
                'a/img.png', stream=io.BytesIO(b''), content_type='image/png'
            )

    @pytest.mark.asyncio
    async def test_write_stream_does_not_swallow_unexpected_errors(
        self, s3_db_client, mock_s3_client
    ):

        mock_s3_client.upload_fileobj.side_effect = RuntimeError('Unexpected')

        with pytest.raises(RuntimeError, match='Unexpected'):
            await s3_db_client.write_stream(
                'a/img.png', stream=io.BytesIO(b''), content_type='image/png'
            )

    @pytest.mark.asyncio
    async def test_delete_object_deletes_object(self, s3_db_client, mock_s3_client):

        await s3_db_client.delete_object('a/b.json')

        mock_s3_client.delete_object.assert_called_once_with(Bucket='bucket', Key='a/b.json')

    @pytest.mark.asyncio
    async def test_delete_object_raises_on_failure(self, s3_db_client, mock_s3_client):

        mock_s3_client.delete_object.side_effect = _client_error('AccessDenied', 'DeleteObject')

        with pytest.raises(DBClientException, match='Failed to delete the object at a/b.json'):
            await s3_db_client.delete_object('a/b.json')

    @pytest.mark.asyncio
    async def test_list_objects_follows_pagination(self, s3_db_client, mock_s3_client):

        paginator = mock_s3_client.get_paginator.return_value
        paginator.paginate.return_value = [
            {'Contents': [{'Key': 'images/1.png'}, {'Key': 'images/2.png'}]},
            {'Contents': [{'Key': 'images/3.png'}]},
            {'KeyCount': 0},
        ]

        assert await s3_db_client.list_objects('images/') == [
            'images/1.png',
            'images/2.png',
            'images/3.png',
        ]

        mock_s3_client.get_paginator.assert_called_once_with('list_objects_v2')
        paginator.paginate.assert_called_once_with(Bucket='bucket', Prefix='images/')

    @pytest.mark.asyncio
    async def test_list_objects_raises_on_failure(self, s3_db_client, mock_s3_client):

        mock_s3_client.get_paginator.return_value.paginate.side_effect = _client_error(
            'AccessDenied', 'ListObjectsV2'
        )

        with pytest.raises(DBClientException, match='Failed to list the objects under images/'):
            await s3_db_client.list_objects('images/')

    @pytest.mark.asyncio
    async def test_list_subdirectories_follows_pagination(self, s3_db_client, mock_s3_client):

        paginator = mock_s3_client.get_paginator.return_value
        paginator.paginate.return_value = [
            {'CommonPrefixes': [{'Prefix': 'surveys/a/'}]},
            {'CommonPrefixes': [{'Prefix': 'surveys/b/'}]},
            {'KeyCount': 0},
        ]

        assert await s3_db_client.list_subdirectories('surveys/') == ['surveys/a/', 'surveys/b/']

        paginator.paginate.assert_called_once_with(
            Bucket='bucket', Prefix='surveys/', Delimiter='/'
        )

    @pytest.mark.asyncio
    async def test_list_subdirectories_raises_on_failure(self, s3_db_client, mock_s3_client):

        mock_s3_client.get_paginator.return_value.paginate.side_effect = _client_error(
            'AccessDenied', 'ListObjectsV2'
        )

        with pytest.raises(
            DBClientException, match='Failed to list the subdirectories of surveys/'
        ):
            await s3_db_client.list_subdirectories('surveys/')
