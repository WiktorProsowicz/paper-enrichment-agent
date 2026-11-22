# Paper Access Agent: tools

Each tool is a lookup in an external bibliographic service; the tools share no state. A tool
error is returned to the agent with a hint on how to correct the call.

## OpenAlex search

State: none.

### T-Search-OpenAlex: `scrape_metadata_from_openalex`

Searches the OpenAlex catalog by title and returns up to `openalex_max_results` compact work
summaries. The agent picks the work matching the signature's authors and year; it is the source of
the DOI, authors, date and arXiv ID used by all other steps.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `title` | `str` | The paper title only, without authors, venue or year. |

**Output**: `list[OpenAlexWork]`

```json
[{"openalex_id": "https://openalex.org/W2626778328", "title": "Attention Is All You Need",
  "doi": "10.48550/arxiv.1706.03762", "publication_date": "2017-06-12",
  "authors": ["Ashish Vaswani", "Noam Shazeer", "..."], "venue": "arXiv (Cornell University)",
  "arxiv_id": "1706.03762", "relevance_score": 300.3}]
```

`doi`, `publication_date`, `venue`, `arxiv_id` and `relevance_score` may be `null`. An empty list
means no match.

**Errors**:
- Empty title, OpenAlex unreachable or an invalid response -> "Pass a non-empty title derived from
  the citation signature, or retry later."

## arXiv HTML check

State: none.

### T-Check-Arxiv-Html: `has_arxiv_html_version`

Checks whether an arXiv paper has an HTML version (on ar5iv), i.e. whether it can be used as the
`arxiv` source. Papers submitted without LaTeX source have none.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `arxiv_id` | `str` | arXiv ID, e.g. `1706.03762` or `hep-th/9901001`. |

**Output**: `bool`, e.g. `true`.

**Errors**:
- Malformed ID, unknown paper (404), unexpected redirect or status, or arXiv unreachable -> the
  reason followed by "Verify the arXiv ID reported by OpenAlex."

## arXiv PDF check

State: none.

### T-Check-Arxiv-Pdf: `is_arxiv_pdf_location_correct`

Checks whether an arXiv PDF URL points to an existing PDF. Used for arXiv papers without an HTML
version; the URL is built by the agent as `https://arxiv.org/pdf/<arxiv_id>`.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `pdf_url` | `str` | The PDF URL, e.g. `https://arxiv.org/pdf/1706.03762`. |

**Output**: `bool`: `true` only if arXiv answers with a PDF (`application/pdf`); a missing file
gives `false`, not an error.

**Errors**:
- The URL does not have the `https://arxiv.org/pdf/<arxiv_id>` form -> the message shows the
  expected form.
- arXiv unreachable -> "Failed to reach arXiv: ...".

## Unpaywall query

State: none.

### T-Query-Unpaywall: `query_unpaywall`

Queries Unpaywall for the open access PDF locations of a paper by its DOI. Used when the paper is
not available on arXiv.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `doi` | `str` | DOI of the paper, e.g. `10.1609/aaai.v34i07.6693`. |

**Output**: `UnpaywallResult`

```json
{"doi": "10.1609/aaai.v34i07.6693", "title": "Channel Attention Is All You Need for Video Frame Interpolation",
 "is_oa": true, "best_pdf_url": null,
 "pdf_locations": [{"url_for_pdf": "https://ojs.aaai.org/index.php/AAAI/article/download/6693/6547",
                    "host_type": "repository", "version": "submittedVersion", "license": null}]}
```

`pdf_locations` are ordered best first; `version` is one of `publishedVersion`,
`acceptedVersion`, `submittedVersion`.

**Errors**:
- Unknown DOI, Unpaywall unreachable or an invalid response -> "Verify the DOI reported by
  OpenAlex."
