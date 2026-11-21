"""Contains the tools the paper access agent uses to find publicly available versions of papers.

The tools delegate to the fetchers of
:mod:`~paper_enrichment_agent.paper_scraping_service.components.fetchers`, which query external
bibliographic services (OpenAlex, arXiv / ar5iv, Unpaywall), and report their failures to the agent
as actionable errors. They are exposed to the agent through an MCP server, see
:meth:`PaperAccessTools.as_mcp_server`.

Exported classes:
    PaperAccessTools: The tools exposed to the paper access agent.
"""

import asyncio
from typing import Annotated, Self

import fastmcp
import httpx
import pydantic
from pydantic import Field

from paper_enrichment_agent.paper_scraping_service.components.fetchers import (
    ArxivMetadataFetcher,
    OpenAlexScraper,
    OpenAlexWork,
    UnpaywallFetcher,
    UnpaywallResult,
)


class PaperAccessTools:
    """The tools exposed to the paper access agent.

    Each public method other than :meth:`create` and :meth:`as_mcp_server` is a tool, whose
    docstring is the tool description seen by the agent. The tools raise `ValueError` with an
    actionable message, which is returned to the agent as the tool error. The tools are stateless
    and async (the synchronous arXiv fetcher is offloaded to a thread), so a single instance may be
    shared between agent invocations.
    """

    class Configuration(pydantic.BaseModel):
        """Configuration of the paper access tools, i.e. of the fetchers they create."""

        openalex_api_url: Annotated[
            str,
            Field(description='The URL of the OpenAlex works endpoint.', min_length=1),
        ]
        openalex_api_key: Annotated[
            pydantic.SecretStr,
            Field(description='The OpenAlex API key. If empty, the API is queried without it.'),
        ]
        openalex_max_results: Annotated[
            int,
            Field(description='The number of OpenAlex works shown to the agent.', ge=1, le=10),
        ]
        unpaywall_api_url: Annotated[
            str, Field(description='The base URL of the Unpaywall v2 API.', min_length=1)
        ]
        unpaywall_email: Annotated[
            str,
            Field(description='The contact email required by Unpaywall.', min_length=1),
        ]
        arxiv_api_url: Annotated[
            str,
            Field(description='The URL of the arXiv export API query endpoint.', min_length=1),
        ]
        ar5iv_html_url_base: Annotated[
            str,
            Field(
                description='The base URL of the ar5iv HTML versions of arXiv papers.',
                min_length=1,
            ),
        ]
        request_timeout: Annotated[
            float, Field(description='The timeout (in seconds) of a single request.', gt=0)
        ]

    def __init__(
        self,
        openalex_scraper: OpenAlexScraper,
        arxiv_fetcher: ArxivMetadataFetcher,
        unpaywall_fetcher: UnpaywallFetcher,
    ) -> None:

        self._openalex_scraper = openalex_scraper
        self._arxiv_fetcher = arxiv_fetcher
        self._unpaywall_fetcher = unpaywall_fetcher

    @classmethod
    def create(cls, config: Configuration, http_client: httpx.AsyncClient) -> Self:
        """Creates the tools along with the fetchers they use.

        Args:
            config: The configuration of the tools.
            http_client: The HTTP client of the OpenAlex and Unpaywall fetchers.

        Returns:
            The created tools.
        """

        return cls(
            openalex_scraper=OpenAlexScraper(
                http_client=http_client,
                api_url=config.openalex_api_url,
                api_key=config.openalex_api_key.get_secret_value(),
                max_results=config.openalex_max_results,
                timeout=config.request_timeout,
            ),
            arxiv_fetcher=ArxivMetadataFetcher(
                api_url=config.arxiv_api_url,
                ar5iv_html_url_base=config.ar5iv_html_url_base,
                timeout=config.request_timeout,
            ),
            unpaywall_fetcher=UnpaywallFetcher(
                http_client=http_client,
                api_url=config.unpaywall_api_url,
                email=config.unpaywall_email,
                timeout=config.request_timeout,
            ),
        )

    def as_mcp_server(self) -> fastmcp.FastMCP:
        """Creates an MCP server exposing the tools.

        Returns:
            The MCP server, which can be connected to in-memory with `fastmcp.Client(server)`.
        """

        mcp_server = fastmcp.FastMCP(
            name='paper_access_tools', on_duplicate='error', strict_input_validation=True
        )

        mcp_server.tool(self.scrape_metadata_from_openalex, name='scrape_metadata_from_openalex')
        mcp_server.tool(self.has_arxiv_html_version, name='has_arxiv_html_version')
        mcp_server.tool(self.is_arxiv_pdf_location_correct, name='is_arxiv_pdf_location_correct')
        mcp_server.tool(self.query_unpaywall, name='query_unpaywall')

        return mcp_server

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
        University)", "arxiv_id": "1706.03762", "relevance_score": 300.3}]

        Args:
            title: The title of the paper, e.g. `Attention is all you need`.
        """

        try:
            return await self._openalex_scraper.search_paper_by_title(title)

        except OpenAlexScraper.FetchingError as e:
            raise ValueError(
                f'{e} Pass a non-empty title derived from the citation signature, or retry later.'
            ) from e

    async def has_arxiv_html_version(self, arxiv_id: str) -> bool:
        """Checks whether an arXiv paper has an HTML version, which can be parsed.

        Use this tool for papers hosted on arXiv. An arXiv paper should be used as the `arxiv`
        source only if this tool returns `true`. Papers submitted to arXiv without their LaTeX
        source have no HTML version.

        Args:
            arxiv_id: The arXiv ID of the paper, e.g. `1706.03762` or `hep-th/9901001`.
        """

        try:
            return await asyncio.to_thread(self._arxiv_fetcher.has_arxiv_html_version, arxiv_id)

        except ArxivMetadataFetcher.FetchingError as e:
            raise ValueError(f'{e} Verify the arXiv ID reported by OpenAlex.') from e

    async def is_arxiv_pdf_location_correct(self, pdf_url: str) -> bool:
        """Checks whether an arXiv PDF URL points to an existing PDF file.

        Use this tool for papers hosted on arXiv, which have no HTML version. Construct the URL
        from the arXiv ID of the paper as `https://arxiv.org/pdf/<arxiv_id>`. Use the URL as the
        `pdf` source only if this tool returns `true`.

        Args:
            pdf_url: The URL of the PDF, e.g. `https://arxiv.org/pdf/1706.03762`.
        """

        try:
            return await asyncio.to_thread(
                self._arxiv_fetcher.is_arxiv_pdf_location_correct, pdf_url
            )

        except ArxivMetadataFetcher.FetchingError as e:
            raise ValueError(str(e)) from e

    async def query_unpaywall(self, doi: str) -> UnpaywallResult:
        """Queries Unpaywall for the publicly available PDF versions of a paper.

        Use this tool when the paper is not available on arXiv. It requires the DOI of the paper,
        e.g. the one reported by OpenAlex. Prefer `best_pdf_url`, otherwise pick one of
        `pdf_locations`, preferring the published versions.

        ## Example output:
        {"doi": "10.1609/aaai.v34i07.6693", "title": "Channel Attention Is All You Need for
        Video Frame Interpolation", "is_oa": true, "best_pdf_url": null, "pdf_locations":
        [{"url_for_pdf": "https://ojs.aaai.org/index.php/AAAI/article/download/6693/6547",
        "host_type": "repository", "version": "submittedVersion", "license": null}]}

        Args:
            doi: The DOI of the paper, e.g. `10.1609/aaai.v34i07.6693`.
        """

        try:
            return await self._unpaywall_fetcher.fetch(doi)

        except UnpaywallFetcher.FetchingError as e:
            raise ValueError(f'{e} Verify the DOI reported by OpenAlex.') from e
