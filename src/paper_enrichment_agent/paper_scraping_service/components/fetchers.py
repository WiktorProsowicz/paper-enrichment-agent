"""Contains fetchers retrieving metadata / open-access sourcees of papers from external services.

See :class:`~paper_enrichment_agent.common.models.misc.PaperMetadata`.

Exported classes:
    ArxivMetadataFetcher: Fetches the metadata of arXiv papers from the arXiv export API.
"""

import datetime
import re
import xml.etree.ElementTree as ET

import requests

from paper_enrichment_agent.common.models import misc as misc_models


class ArxivMetadataFetcher:
    """Fetches the metadata of arXiv papers from the arXiv export API.

    The API responds with an Atom feed (see https://info.arxiv.org/help/api/user-manual.html),
    from which the authors, the date of the first version and the DOI are read. The DOI declared by
    the authors (usually the one of the journal version) is preferred. Otherwise, the DOI assigned
    to the paper by arXiv (`10.48550/arXiv.<id>`) is used.

    The fetcher is synchronous and stateless, so a single instance can be shared, also between
    threads. It raises `FetchingError` if the metadata cannot be downloaded or read.
    """

    class FetchingError(Exception):
        """Raised when the metadata of a paper cannot be fetched."""

    # Maps the prefixes used in the element queries to the XML namespaces of the API response:
    # the standard Atom elements (`entry`, `author`, `published`) and arXiv's extensions (`doi`).
    _NAMESPACES = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom',
    }

    # Prefix of the DOIs registered by arXiv for its papers
    _ARXIV_DOI_PREFIX = '10.48550/arXiv.'

    # Matches the version suffix of an arXiv ID, e.g. `v2` in `1706.03762v2`
    _VERSION_SUFFIX_PATTERN = re.compile(r'v\d+$')

    def __init__(self, api_url: str, timeout: float) -> None:
        """Initializes the fetcher.

        Args:
            api_url: The URL of the arXiv export API query endpoint.
            timeout: The timeout (in seconds) of a single API request.
        """

        self._api_url = api_url
        self._timeout = timeout

    def fetch(self, arxiv_id: str) -> misc_models.PaperMetadata:
        """Downloads the metadata of an arXiv paper.

        Args:
            arxiv_id: The arXiv ID of the paper, optionally with a version suffix.

        Returns:
            The metadata of the paper.

        Raises:
            FetchingError: If the request fails, or the response does not describe the paper.
        """

        try:
            response = requests.get(
                self._api_url, params={'id_list': arxiv_id}, timeout=self._timeout
            )
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise self.FetchingError(
                f'Failed to download the metadata of arXiv paper {arxiv_id}: {e}'
            ) from e

        try:
            feed = ET.fromstring(response.content)

        except ET.ParseError as e:
            raise self.FetchingError(
                f'Failed to parse the metadata response for arXiv paper {arxiv_id}: {e}'
            ) from e

        return self._read_entry(feed, arxiv_id)

    def _read_entry(self, feed: ET.Element, arxiv_id: str) -> misc_models.PaperMetadata:
        """Reads the metadata from the single entry of the API response feed.

        Raises:
            FetchingError: If the feed contains no valid entry for the paper.
        """

        entry = feed.find('atom:entry', self._NAMESPACES)
        entry_id = entry.findtext('atom:id', '', self._NAMESPACES) if entry is not None else ''

        # The API reports invalid IDs as an entry pointing to `http://arxiv.org/api/errors`.
        if entry is None or not entry_id or '/api/errors' in entry_id:
            raise self.FetchingError(f'The arXiv API returned no paper for ID {arxiv_id}.')

        authors = [
            name.strip()
            for author in entry.findall('atom:author', self._NAMESPACES)
            if (name := author.findtext('atom:name', '', self._NAMESPACES)).strip()
        ]

        published = entry.findtext('atom:published', '', self._NAMESPACES).strip()

        try:
            publication_date = datetime.datetime.fromisoformat(published).date()

        except ValueError as e:
            raise self.FetchingError(
                f'Invalid publication date "{published}" of arXiv paper {arxiv_id}.'
            ) from e

        doi = entry.findtext('arxiv:doi', '', self._NAMESPACES).strip()

        if not doi:
            unversioned_id = self._VERSION_SUFFIX_PATTERN.sub('', arxiv_id)
            doi = f'{self._ARXIV_DOI_PREFIX}{unversioned_id}'

        return misc_models.PaperMetadata(
            doi=doi, authors=authors, publication_date=publication_date
        )
