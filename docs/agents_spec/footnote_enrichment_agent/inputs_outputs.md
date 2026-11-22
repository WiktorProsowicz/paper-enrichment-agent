# Footnote Enrichment Agent: inputs and outputs

## Inputs

### Specification

Model: `FootnoteEnrichmentRequest`

```json
{
  "session_id": "<str: id of the enrichment session, also the tracing session; random by default>",
  "survey_title": "<str: title of the citing survey>",
  "survey_abstract": "<str: abstract of the citing survey>",
  "reference_document": "<Document: the cited document, parsed into the document tree>",
  "reference_id": "<str: component_id of the citation Reference inside referencing_paragraph>",
  "referencing_paragraph": "<Paragraph: the survey paragraph containing the citation>",
  "citation_context_info": "<str: free-form description of where the citation occurs in the survey>"
}
```

`citation_context_info` is the chain of section titles leading to the citation, e.g. `The
document was cited in the section specified by the following path in the document tree:
'Acoustic models' > 'Autoregressive models'`.

| Prompt variable | Source |
| --- | --- |
| `survey_title`, `survey_abstract` | The request fields. |
| `cited_doc_title`, `cited_doc_abstract` | The title and the abstract of `reference_document`. |
| `cited_doc_toc` | The tree of contents of `reference_document`: the type, path and id of every component, with section titles and figure captions. |
| `citation_reference_text` | The text of the citation marker identified by `reference_id`, e.g. `42`. |
| `citation_context_fragment` | Up to `citation_context_chars` characters of `referencing_paragraph` around the citation. |
| `citation_context_info` | The request field. |

### Examples

#### IN-Detailed-Citation

The survey describes the cited work in some detail, so the footnote should explain exactly the
described parts.

```json
{
  "survey_title": "A Survey on Neural Speech Synthesis",
  "reference_id": "ref_42",
  "referencing_paragraph": {"elements": [
    "... the authors of ",
    {"ref_type": "citation", "component_id": "ref_42", "content_text": "42", "target": "bib_42"},
    " propose a convolutional network that generates the next spectrogram frame conditioned on the speaker embedding ..."
  ]},
  "citation_context_info": "The document was cited in the section specified by the following path in the document tree: 'Acoustic models' > 'CNN-based models'"
}
```

#### IN-Group-Citation

The citation is one of several in a generic statement, so the footnote should only briefly
describe the cited work's methodology.

```json
{
  "referencing_paragraph": {"elements": [
    "To date, many TTS systems applied classical regression to generate speech [",
    {"ref_type": "citation", "component_id": "ref_1", "content_text": "1", "target": "bib_1"},
    {"ref_type": "citation", "component_id": "ref_2", "content_text": "2", "target": "bib_2"},
    "]."
  ]},
  "reference_id": "ref_2"
}
```

## Outputs

### Specification

The agent's answer is the **prelude** of the footnote: free-form text, the LLM's final message.
It is a concise, abstract-like introduction to the part of the cited document that was extracted
into the footnote. It contains only the prelude: no reasoning, no report of the actions taken, no
messages addressed to the user.

The other kind of output is the **footnote** itself, produced through the state of the tools (see
[tools.md](tools.md#enrichment-state)). It is a `Section`:

```json
{
  "component_id": "<str: generated id>",
  "title": "<str: set by T-Set-Footnote-Title, empty if never set>",
  "components": [
    "<Paragraph: a whole paragraph or a citation fragment>",
    "<Figure: with all or the selected subfigures>",
    "<MathExpression: a standalone equation>",
    "<Section: mirrors a source section above drop_sections_from_level, with its id and title, holding the components above>"
  ]
}
```

### Examples

#### OUT-Focused-Prelude

Produced for `IN-Detailed-Citation`: the prelude introduces the specific part of the cited work
the survey describes.

```
The paper introduces a convolutional acoustic model that predicts each spectrogram frame from the
previous ones, conditioned on a learned speaker embedding that lets a single model voice many
speakers.
```

#### OUT-Overview-Prelude

Produced for `IN-Group-Citation`: the prelude briefly summarizes the cited work's overall
methodology.

```
The work trains a feed-forward network to regress mel-spectrogram frames from linguistic
features, minimizing the L1 distance between the predicted and the target frames.
```
