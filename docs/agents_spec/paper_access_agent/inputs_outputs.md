# Paper Access Agent: inputs and outputs

## Inputs

### Specification

A single non-empty free-form string: the **citation signature** of the paper, as found in the
bibliography of the citing paper. It usually contains the authors, the title, the venue and the
year, in a style that depends on the citing paper.

| Prompt variable | Source |
| --- | --- |
| `citation_signature` | The input. |
| `output_schema` | The JSON schema of `PaperAccessInfo`. |

### Examples

#### IN-Full-Signature

Authors, title, venue and year are present.

```
A. Vaswani, N. Shazeer, et al. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.
```

#### IN-Abbreviated-Signature

Abbreviated venue, truncated author list; the title must still be derived from it.

```
Huang et al., "Channel attention is all you need for video frame interpolation," AAAI, 2020.
```

#### IN-Unresolvable-Signature

The signature describes something without a public version or without a catalog entry (a book,
a technical report, a malformed signature).

```
J. Smith. Internal technical report on speech corpora. ACME Corp., 1998.
```

## Outputs

### Specification

Model: `PaperAccessInfo | None`. The final message must be exactly this JSON, or `null`.

```json
{
  "type": "<'arxiv' | 'pdf': kind of the public source>",
  "source": "<str: arXiv ID for 'arxiv', URL of the PDF for 'pdf'>",
  "metadata": {
    "doi": "<str: DOI of the published version, else the arXiv DOI 10.48550/arXiv.<id>>",
    "authors": ["<str: full names, in publication order>"],
    "publication_date": "<date YYYY-MM-DD: first publication>"
  }
}
```

### Examples

#### OUT-Arxiv-Source

The paper is on arXiv and has an HTML version: the source is its arXiv ID.

```json
{"type": "arxiv", "source": "1706.03762", "metadata": {"doi": "10.48550/arXiv.1706.03762", "authors": ["Ashish Vaswani", "Noam Shazeer", "..."], "publication_date": "2017-06-12"}}
```

#### OUT-Pdf-Source

The paper has no arXiv HTML version but an open access PDF: the source is the PDF URL, either
the arXiv PDF of a paper without LaTeX source, or a location reported by Unpaywall.

```json
{"type": "pdf", "source": "https://arxiv.org/pdf/2101.00001", "metadata": {"doi": "10.48550/arXiv.2101.00001", "authors": ["..."], "publication_date": "2021-01-01"}}
```

```json
{"type": "pdf", "source": "https://ojs.aaai.org/index.php/AAAI/article/download/6693/6547", "metadata": {"doi": "10.1609/aaai.v34i07.6693", "authors": ["Myungsub Choi", "..."], "publication_date": "2020-04-03"}}
```

#### OUT-Not-Found

No OpenAlex work matches the signature, or the paper has no public version.

```json
null
```
