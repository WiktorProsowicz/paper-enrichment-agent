"""Contains the tools the paper access agent uses to find publicly available versions of papers.

The tools query external bibliographic services (OpenAlex, ar5iv, Unpaywall) and pre-process their
responses into compact summaries, which are shown to the LLM agent. They are exposed to the agent
through an MCP server, see :func:`create_mcp_server`.

Exported classes:
    OpenAlexWork: Summary of a single OpenAlex work matching a searched title.
    OpenAccessLocation: A single open access location of a paper reported by Unpaywall.
    UnpaywallResult: Summary of the open access status of a paper reported by Unpaywall.
    PaperAccessTools: The tools exposed to the paper access agent.
"""

import re
from typing import Annotated, Any

import fastmcp
import httpx
import pydantic
from pydantic import Field


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
    is_oa: Annotated[bool, Field(description='Whether the work is open access.')]
    oa_url: Annotated[
        str | None, Field(description='The URL of the best open access version, if any.')
    ]
    relevance_score: Annotated[
        float | None, Field(description='The relevance of the work to the searched title.')
    ]


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


class PaperAccessTools:
    """The tools exposed to the paper access agent.

    Each public method is a tool, whose docstring is the tool description seen by the agent. The
    tools raise `ValueError` with an actionable message, which is returned to the agent as the
    tool error. The tools are stateless and async, the injected HTTP client may be shared.
    """

    class Configuration(pydantic.BaseModel):
        """Configuration of the paper access tools."""

        openalex_api_url: Annotated[
            str,
            Field(description='The URL of the OpenAlex works endpoint.', min_length=1),
        ] = 'https://api.openalex.org/works'
        openalex_api_key: Annotated[
            pydantic.SecretStr,
            Field(description='The OpenAlex API key. If empty, the API is queried without it.'),
        ] = pydantic.SecretStr('')
        openalex_max_results: Annotated[
            int,
            Field(description='The number of OpenAlex works shown to the agent.', ge=1, le=10),
        ] = 3
        unpaywall_api_url: Annotated[
            str, Field(description='The base URL of the Unpaywall v2 API.', min_length=1)
        ] = 'https://api.unpaywall.org/v2'
        unpaywall_email: Annotated[
            str,
            Field(description='The contact email required by Unpaywall.', min_length=1),
        ]
        ar5iv_html_url_base: Annotated[
            str,
            Field(description='The base URL of the ar5iv HTML versions of arXiv papers.'),
        ] = 'https://ar5iv.labs.arxiv.org/html'
        request_timeout: Annotated[
            float, Field(description='The timeout (in seconds) of a single request.', gt=0)
        ] = 30.0

    # Fields of the OpenAlex works read by the tools
    _OPENALEX_SELECTED_FIELDS = (
        'id',
        'doi',
        'title',
        'publication_date',
        'authorships',
        'primary_location',
        'locations',
        'open_access',
        'relevance_score',
    )

    # New-style (`1706.03762`) and old-style (`hep-th/9901001`) arXiv IDs, optionally versioned
    _ARXIV_ID_PATTERN = r'(?:\d{4}\.\d{4,5}|[a-z]+(?:-[a-z]+)*(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?'
    _ARXIV_ID_REGEX = re.compile(rf'^{_ARXIV_ID_PATTERN}$')
    _ARXIV_URL_REGEX = re.compile(rf'arxiv\.org/(?:abs|pdf)/({_ARXIV_ID_PATTERN})', re.IGNORECASE)
    _ARXIV_DOI_REGEX = re.compile(rf'^10\.48550/arxiv\.({_ARXIV_ID_PATTERN})$', re.IGNORECASE)
    _VERSION_SUFFIX_REGEX = re.compile(r'v\d+$')

    _DOI_URL_PREFIX = 'https://doi.org/'

    # Marker of the LaTeXML-generated article pages served by ar5iv
    _AR5IV_ARTICLE_MARKER = 'ltx_document'

    def __init__(self, config: Configuration, http_client: httpx.AsyncClient) -> None:

        self._config = config
        self._http_client = http_client

    async def scrape_metadata_from_openalex(self, title: str) -> list[OpenAlexWork]:
        """Searches the OpenAlex catalog for the works matching a paper title.

        Use this tool first, to identify the cited paper and obtain its DOI, authors, publication
        date and arXiv ID. Pass only the title of the paper, derived from the citation signature,
        without the authors, venue or year. Compare the authors and the year of the returned works
        with the citation to pick the right one. If none of them matches, retry with a corrected
        or shortened title.

        ## Example output:
        [{"openalex_id": "https://openalex.org/W2626778328", "title": "Attention Is All You
        Need", "doi": "10.48550/arxiv.1706.03762", "publication_date": "2017-06-12",
        "authors": ["Ashish Vaswani", "Noam Shazeer", ...], "venue": "arXiv (Cornell
        University)", "arxiv_id": "1706.03762", "is_oa": true, "oa_url":
        "https://arxiv.org/pdf/1706.03762", "relevance_score": 300.3}]

        Args:
            title: The title of the paper, e.g. `Attention is all you need`.
        """

        if not title.strip():
            raise ValueError('The title must not be empty. Derive it from the citation signature.')

        params: dict[str, Any] = {
            'search': title.strip(),
            'per_page': self._config.openalex_max_results,
            'select': ','.join(self._OPENALEX_SELECTED_FIELDS),
        }

        if api_key := self._config.openalex_api_key.get_secret_value():
            params['api_key'] = api_key

        response_json = await self._get_json(
            self._config.openalex_api_url, params=params, service_name='OpenAlex'
        )

        return [
            self._summarize_openalex_work(work)
            for work in response_json.get('results', [])[: self._config.openalex_max_results]
        ]

    async def has_arxiv_html_version(self, arxiv_id: str) -> bool:
        """Checks whether an arXiv paper has an HTML version, which can be parsed.

        Use this tool for papers hosted on arXiv. An arXiv paper should be preferred over a PDF
        only if this tool returns `true`. Papers submitted to arXiv without their LaTeX source
        have no HTML version.

        Args:
            arxiv_id: The arXiv ID of the paper, e.g. `1706.03762` or `hep-th/9901001`.
        """

        arxiv_id = arxiv_id.strip().removeprefix('arXiv:')

        if not self._ARXIV_ID_REGEX.match(arxiv_id):
            raise ValueError(
                f'"{arxiv_id}" is not a valid arXiv ID. Use the format `1706.03762` or '
                '`hep-th/9901001`, e.g. the `arxiv_id` reported by OpenAlex.'
            )

        try:
            response = await self._http_client.get(
                f'{self._config.ar5iv_html_url_base}/{arxiv_id}',
                follow_redirects=False,
                timeout=self._config.request_timeout,
            )

        except httpx.HTTPError as e:
            raise ValueError(f'Failed to reach ar5iv: {e}. Retry the call later.') from e

        if response.is_redirect:
            if '/abs/' in response.headers.get('location', ''):
                return False

            raise ValueError(
                f'ar5iv redirected the paper {arxiv_id} to an unexpected location '
                f'"{response.headers.get("location")}".'
            )

        if response.status_code == httpx.codes.NOT_FOUND:
            raise ValueError(
                f'The arXiv paper {arxiv_id} does not exist. Verify the arXiv ID with OpenAlex.'
            )

        if not response.is_success:
            raise ValueError(
                f'ar5iv responded with status {response.status_code} for the paper {arxiv_id}.'
            )

        return (
            'text/html' in response.headers.get('content-type', '')
            and self._AR5IV_ARTICLE_MARKER in response.text
        )

    async def query_unpaywall(self, doi: str) -> UnpaywallResult:
        """Queries Unpaywall for the publicly available PDF versions of a paper.

        Use this tool when the paper has no HTML version on arXiv. It requires the DOI of the
        paper, e.g. the one reported by OpenAlex. Prefer `best_pdf_url`, otherwise pick one of
        `pdf_locations`, preferring the published versions.

        ## Example output:
        {"doi": "10.1609/aaai.v34i07.6693", "title": "Channel Attention Is All You Need for
        Video Frame Interpolation", "is_oa": true, "best_pdf_url": null, "pdf_locations":
        [{"url_for_pdf": "https://ojs.aaai.org/index.php/AAAI/article/download/6693/6547",
        "host_type": "repository", "version": "submittedVersion", "license": null}]}

        Args:
            doi: The DOI of the paper, e.g. `10.1609/aaai.v34i07.6693`.
        """

        doi = doi.strip().removeprefix(self._DOI_URL_PREFIX)

        if not doi.startswith('10.'):
            raise ValueError(f'"{doi}" is not a valid DOI. A DOI starts with `10.`.')

        response_json = await self._get_json(
            f'{self._config.unpaywall_api_url}/{doi}',
            params={'email': self._config.unpaywall_email},
            service_name='Unpaywall',
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

    async def _get_json(
        self, url: str, params: dict[str, Any], service_name: str
    ) -> dict[str, Any]:
        """Sends a GET request and decodes the JSON response, reporting errors to the agent."""

        try:
            response = await self._http_client.get(
                url, params=params, timeout=self._config.request_timeout
            )

        except httpx.HTTPError as e:
            raise ValueError(f'Failed to reach {service_name}: {e}. Retry the call later.') from e

        if response.status_code == httpx.codes.NOT_FOUND:
            raise ValueError(
                f'{service_name} has no record for the request. Verify the passed identifier.'
            )

        if not response.is_success:
            raise ValueError(
                f'{service_name} responded with status {response.status_code}: '
                f'{response.text[:200]}'
            )

        try:
            response_json = response.json()

        except ValueError as e:
            raise ValueError(f'{service_name} returned an invalid JSON response.') from e

        if not isinstance(response_json, dict):
            raise ValueError(f'{service_name} returned an unexpected response.')

        return response_json

    def _summarize_openalex_work(self, work: dict[str, Any]) -> OpenAlexWork:
        """Converts a raw OpenAlex work into the summary shown to the agent."""

        doi = (work.get('doi') or '').removeprefix(self._DOI_URL_PREFIX) or None
        primary_source = (work.get('primary_location') or {}).get('source') or {}
        open_access = work.get('open_access') or {}

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
            is_oa=bool(open_access.get('is_oa')),
            oa_url=open_access.get('oa_url'),
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


def create_mcp_server(tools: PaperAccessTools) -> fastmcp.FastMCP:
    """Creates an MCP server exposing the paper access tools.

    Args:
        tools: The tools to expose.

    Returns:
        The MCP server, which can be connected to in-memory with `fastmcp.Client(server)`.
    """

    mcp_server = fastmcp.FastMCP(
        name='paper_access_tools', on_duplicate='error', strict_input_validation=True
    )

    mcp_server.tool(tools.scrape_metadata_from_openalex, name='scrape_metadata_from_openalex')
    mcp_server.tool(tools.has_arxiv_html_version, name='has_arxiv_html_version')
    mcp_server.tool(tools.query_unpaywall, name='query_unpaywall')

    return mcp_server
