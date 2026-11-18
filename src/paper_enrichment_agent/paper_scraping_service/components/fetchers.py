"""Contains fetchers retrieving metadata / open-access sources of papers from external services.

See :class:`~paper_enrichment_agent.common.models.misc.PaperMetadata`.

Exported classes:
    ArxivMetadataFetcher: Fetches the metadata of arXiv papers and checks their public versions.
    OpenAlexWork: Summary of a single OpenAlex work matching a searched title.
    OpenAlexScraper: Searches the OpenAlex catalog for papers.
    OpenAccessLocation: A single open access location of a paper reported by Unpaywall.
    UnpaywallResult: Summary of the open access status of a paper reported by Unpaywall.
    UnpaywallFetcher: Fetches the open access locations of papers from Unpaywall.
"""

import datetime
import re
import xml.etree.ElementTree as ET
from typing import Annotated, Any

import httpx
import pydantic
import requests
from pydantic import Field

from paper_enrichment_agent.common.models import misc as misc_models

# New-style (`1706.03762`) and old-style (`hep-th/9901001`) arXiv IDs, optionally versioned. It is
# defined above the classes, since their class-level patterns are compiled from it.
_ARXIV_ID_PATTERN = r'(?:\d{4}\.\d{4,5}|[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?'


class ArxivMetadataFetcher:
    """Fetches the metadata of arXiv papers and checks the availability of their public versions.

    The metadata is read from the arXiv export API, which responds with an Atom feed (see
    https://info.arxiv.org/help/api/user-manual.html), from which the authors, the date of the
    first version and the DOI are read. The DOI declared by the authors (usually the one of the
    journal version) is preferred. Otherwise, the DOI assigned to the paper by arXiv
    (`10.48550/arXiv.<id>`) is used.

    The fetcher also checks whether a paper has an HTML version on ar5iv and whether an arXiv PDF
    URL points to an existing PDF.

    The fetcher is synchronous and stateless, so a single instance can be shared, also between
    threads. It raises `FetchingError` if the requested information cannot be downloaded or read.
    """

    class FetchingError(Exception):
        """Raised when the information about a paper cannot be fetched."""

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

    _ARXIV_ID_REGEX = re.compile(rf'^{_ARXIV_ID_PATTERN}$')

    # The only accepted form of the arXiv PDF URLs
    _ARXIV_PDF_URL_REGEX = re.compile(rf'^https://arxiv\.org/pdf/{_ARXIV_ID_PATTERN}$')

    # Marker of the LaTeXML-generated article pages served by ar5iv
    _AR5IV_ARTICLE_MARKER = 'ltx_document'

    def __init__(self, api_url: str, ar5iv_html_url_base: str, timeout: float) -> None:
        """Initializes the fetcher.

        Args:
            api_url: The URL of the arXiv export API query endpoint.
            ar5iv_html_url_base: The base URL of the ar5iv HTML versions of arXiv papers.
            timeout: The timeout (in seconds) of a single request.
        """

        self._api_url = api_url
        self._ar5iv_html_url_base = ar5iv_html_url_base
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

    def has_arxiv_html_version(self, arxiv_id: str) -> bool:
        """Checks whether an arXiv paper has an HTML version on ar5iv.

        ar5iv redirects the papers without an HTML version (e.g. submitted without their LaTeX
        source) to their arXiv abstract pages.

        Args:
            arxiv_id: The arXiv ID of the paper, optionally prefixed with `arXiv:`.

        Returns:
            Whether ar5iv serves an article page of the paper.

        Raises:
            FetchingError: If the ID is invalid, the paper does not exist, or ar5iv responds
                unexpectedly.
        """

        arxiv_id = self._validate_arxiv_id(arxiv_id)

        try:
            response = requests.get(
                f'{self._ar5iv_html_url_base}/{arxiv_id}',
                allow_redirects=False,
                timeout=self._timeout,
            )

        except requests.exceptions.RequestException as e:
            raise self.FetchingError(f'Failed to reach ar5iv: {e}') from e

        if response.is_redirect:
            if '/abs/' in response.headers.get('location', ''):
                return False

            raise self.FetchingError(
                f'ar5iv redirected the paper {arxiv_id} to an unexpected location '
                f'"{response.headers.get("location")}".'
            )

        if response.status_code == requests.codes.not_found:
            raise self.FetchingError(f'The arXiv paper {arxiv_id} does not exist.')

        if not response.ok:
            raise self.FetchingError(
                f'ar5iv responded with status {response.status_code} for the paper {arxiv_id}.'
            )

        return (
            'text/html' in response.headers.get('content-type', '')
            and self._AR5IV_ARTICLE_MARKER in response.text
        )

    def is_arxiv_pdf_location_correct(self, pdf_url: str) -> bool:
        """Checks whether an arXiv PDF URL points to an existing PDF file.

        Args:
            pdf_url: The URL of the PDF, in the `https://arxiv.org/pdf/<arxiv_id>` form.

        Returns:
            Whether arXiv serves a PDF file under the URL.

        Raises:
            FetchingError: If the URL is not an arXiv PDF URL, or arXiv cannot be reached.
        """

        pdf_url = pdf_url.strip()

        if not self._ARXIV_PDF_URL_REGEX.match(pdf_url):
            raise self.FetchingError(
                f'"{pdf_url}" is not an arXiv PDF URL. It must have the form '
                '`https://arxiv.org/pdf/<arxiv_id>`, e.g. `https://arxiv.org/pdf/1706.03762`.'
            )

        try:
            response = requests.head(pdf_url, allow_redirects=True, timeout=self._timeout)

        except requests.exceptions.RequestException as e:
            raise self.FetchingError(f'Failed to reach arXiv: {e}') from e

        return response.ok and 'application/pdf' in response.headers.get('content-type', '')

    def _validate_arxiv_id(self, arxiv_id: str) -> str:
        """Normalizes an arXiv ID, raising `FetchingError` if it is invalid."""

        arxiv_id = arxiv_id.strip().removeprefix('arXiv:')

        if not self._ARXIV_ID_REGEX.match(arxiv_id):
            raise self.FetchingError(
                f'"{arxiv_id}" is not a valid arXiv ID. Use the format `1706.03762` or '
                '`hep-th/9901001`.'
            )

        return arxiv_id

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


class OpenAlexWork(pydantic.BaseModel):
    """Summary of a single OpenAlex work matching a searched title."""

    openalex_id: Annotated[str, Field(description='The OpenAlex ID of the work.')]
    title: Annotated[str, Field(description='The title of the work.')]
    doi: Annotated[str | None, Field(description='The DOI of the work, without the URL prefix.')]
    publication_date: Annotated[
        str | None, Field(description='The publication date in the `YYYY-MM-DD` format.')
    ]
    authors: Annotated[list[str], Field(description='The authors, in the publication order.')]
    venue: Annotated[
        str | None, Field(description='The journal, conference or repository of the work.')
    ]
    arxiv_id: Annotated[
        str | None, Field(description='The arXiv ID of the work, if it is hosted on arXiv.')
    ]
    relevance_score: Annotated[
        float | None, Field(description='The relevance of the work to the searched title.')
    ]


class OpenAlexScraper:
    """Searches the OpenAlex catalog (https://docs.openalex.org) for papers.

    The raw OpenAlex works are pre-processed into compact :class:`OpenAlexWork` summaries. The
    scraper is async and stateless, the injected HTTP client may be shared. It raises
    `FetchingError` if the search fails.
    """

    class FetchingError(Exception):
        """Raised when the OpenAlex catalog cannot be searched."""

    # Fields of the OpenAlex works read by the scraper
    _OPENALEX_SELECTED_FIELDS = (
        'id',
        'doi',
        'title',
        'publication_date',
        'authorships',
        'primary_location',
        'locations',
        'relevance_score',
    )

    _ARXIV_URL_REGEX = re.compile(rf'arxiv\.org/(?:abs|pdf)/({_ARXIV_ID_PATTERN})', re.IGNORECASE)
    _ARXIV_DOI_REGEX = re.compile(rf'^10\.48550/arxiv\.({_ARXIV_ID_PATTERN})$', re.IGNORECASE)

    # Matches the version suffix of an arXiv ID, e.g. `v2` in `1706.03762v2`
    _VERSION_SUFFIX_REGEX = re.compile(r'v\d+$')

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        api_url: str,
        api_key: str,
        max_results: int,
        timeout: float,
    ) -> None:
        """Initializes the scraper.

        Args:
            http_client: The HTTP client sending the requests.
            api_url: The URL of the OpenAlex works endpoint.
            api_key: The OpenAlex API key. If empty, the API is queried without it.
            max_results: The maximum number of returned works.
            timeout: The timeout (in seconds) of a single request.
        """

        self._http_client = http_client
        self._api_url = api_url
        self._api_key = api_key
        self._max_results = max_results
        self._timeout = timeout

    async def search_paper_by_title(self, title: str) -> list[OpenAlexWork]:
        """Searches the OpenAlex catalog for the works matching a paper title.

        Args:
            title: The title of the paper.

        Returns:
            The best matching works, most relevant first.

        Raises:
            FetchingError: If the title is empty, or the search request fails.
        """

        if not title.strip():
            raise self.FetchingError('The searched title must not be empty.')

        params: dict[str, Any] = {
            'search': title.strip(),
            'per_page': self._max_results,
            'select': ','.join(self._OPENALEX_SELECTED_FIELDS),
        }

        if self._api_key:
            params['api_key'] = self._api_key

        response_json = await _get_json(
            self._http_client,
            self._api_url,
            params=params,
            timeout=self._timeout,
            service_name='OpenAlex',
            error_cls=self.FetchingError,
        )

        return [
            self._summarize_work(work)
            for work in response_json.get('results', [])[: self._max_results]
        ]

    def _summarize_work(self, work: dict[str, Any]) -> OpenAlexWork:
        """Converts a raw OpenAlex work into its summary."""

        doi = (work.get('doi') or '').removeprefix(_DOI_URL_PREFIX) or None
        primary_source = (work.get('primary_location') or {}).get('source') or {}

        return OpenAlexWork(
            openalex_id=work.get('id') or '',
            title=work.get('title') or '',
            doi=doi,
            publication_date=work.get('publication_date'),
            authors=[
                name
                for authorship in work.get('authorships') or []
                if (name := (authorship.get('author') or {}).get('display_name'))
            ],
            venue=primary_source.get('display_name'),
            arxiv_id=self._find_arxiv_id(work, doi),
            relevance_score=work.get('relevance_score'),
        )

    def _find_arxiv_id(self, work: dict[str, Any], doi: str | None) -> str | None:
        """Finds the unversioned arXiv ID of a work in its DOI or the URLs of its locations."""

        if doi and (match := self._ARXIV_DOI_REGEX.match(doi)):
            return self._VERSION_SUFFIX_REGEX.sub('', match.group(1))

        for location in work.get('locations') or []:
            for url in (location.get('landing_page_url'), location.get('pdf_url')):
                if url and (match := self._ARXIV_URL_REGEX.search(url)):
                    return self._VERSION_SUFFIX_REGEX.sub('', match.group(1))

        return None


class OpenAccessLocation(pydantic.BaseModel):
    """A single open access location of a paper reported by Unpaywall."""

    url_for_pdf: Annotated[str, Field(description='The URL of the PDF file.')]
    host_type: Annotated[
        str | None, Field(description='The type of the host, `publisher` or `repository`.')
    ]
    version: Annotated[
        str | None,
        Field(
            description=(
                'The version of the paper, e.g. `publishedVersion`, `acceptedVersion`, '
                '`submittedVersion`.'
            )
        ),
    ]
    license: Annotated[str | None, Field(description='The license of the hosted version.')]


class UnpaywallResult(pydantic.BaseModel):
    """Summary of the open access status of a paper reported by Unpaywall."""

    doi: Annotated[str, Field(description='The queried DOI.')]
    title: Annotated[str | None, Field(description='The title of the paper.')]
    is_oa: Annotated[bool, Field(description='Whether the paper is open access.')]
    best_pdf_url: Annotated[
        str | None,
        Field(description='The URL of the PDF of the best open access location, if any.'),
    ]
    pdf_locations: Annotated[
        list[OpenAccessLocation],
        Field(description='All open access locations providing a PDF, best first.'),
    ]


class UnpaywallFetcher:
    """Fetches the open access locations of papers from Unpaywall (https://unpaywall.org).

    The best open access location reported by Unpaywall may lack a PDF, so all locations providing
    a PDF are returned. The fetcher is async and stateless, the injected HTTP client may be shared.
    It raises `FetchingError` if the open access status cannot be fetched.
    """

    class FetchingError(Exception):
        """Raised when the open access status of a paper cannot be fetched."""

    def __init__(
        self, http_client: httpx.AsyncClient, api_url: str, email: str, timeout: float
    ) -> None:
        """Initializes the fetcher.

        Args:
            http_client: The HTTP client sending the requests.
            api_url: The base URL of the Unpaywall v2 API.
            email: The contact email required by Unpaywall.
            timeout: The timeout (in seconds) of a single request.
        """

        self._http_client = http_client
        self._api_url = api_url
        self._email = email
        self._timeout = timeout

    async def fetch(self, doi: str) -> UnpaywallResult:
        """Fetches the open access status and the PDF locations of a paper.

        Args:
            doi: The DOI of the paper, optionally prefixed with `https://doi.org/`.

        Returns:
            The open access status of the paper.

        Raises:
            FetchingError: If the DOI is invalid or unknown to Unpaywall, or the request fails.
        """

        doi = doi.strip().removeprefix(_DOI_URL_PREFIX)

        if not doi.startswith('10.'):
            raise self.FetchingError(f'"{doi}" is not a valid DOI. A DOI starts with `10.`.')

        response_json = await _get_json(
            self._http_client,
            f'{self._api_url}/{doi}',
            params={'email': self._email},
            timeout=self._timeout,
            service_name='Unpaywall',
            error_cls=self.FetchingError,
        )

        best_location = response_json.get('best_oa_location') or {}

        return UnpaywallResult(
            doi=doi,
            title=response_json.get('title'),
            is_oa=bool(response_json.get('is_oa')),
            best_pdf_url=best_location.get('url_for_pdf'),
            pdf_locations=[
                OpenAccessLocation(
                    url_for_pdf=location['url_for_pdf'],
                    host_type=location.get('host_type'),
                    version=location.get('version'),
                    license=location.get('license'),
                )
                for location in response_json.get('oa_locations') or []
                if location.get('url_for_pdf')
            ],
        )


# Prefix of the DOIs given as URLs
_DOI_URL_PREFIX = 'https://doi.org/'


async def _get_json(
    http_client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any],
    timeout: float,
    service_name: str,
    error_cls: type[Exception],
) -> dict[str, Any]:
    """Sends a GET request and decodes the JSON object response, raising `error_cls` on failure."""

    try:
        response = await http_client.get(url, params=params, timeout=timeout)

    except httpx.HTTPError as e:
        raise error_cls(f'Failed to reach {service_name}: {e}') from e

    if response.status_code == httpx.codes.NOT_FOUND:
        raise error_cls(f'{service_name} has no record for the request.')

    if not response.is_success:
        raise error_cls(
            f'{service_name} responded with status {response.status_code}: {response.text[:200]}'
        )

    try:
        response_json = response.json()

    except ValueError as e:
        raise error_cls(f'{service_name} returned an invalid JSON response.') from e

    if not isinstance(response_json, dict):
        raise error_cls(f'{service_name} returned an unexpected response.')

    return response_json
