# Footnote Enrichment Agent: tools

All components are addressed by **paths**: `/sections/<id>/<id>/...` or `/footnotes/<id>/...`,
made of component ids only, as listed in the tree of contents in the user prompt. An unknown path
makes every tool fail with `Component not found in the document: <path>`. A tool error is
returned to the agent, which can correct the call.

## Enrichment state

State: one per enrichment session, created from the request's `reference_document` when the
session starts and discarded with it. It holds:

- the **cited document** (read-only);
- the **footnote title** (empty initially), written by `T-Set-Footnote-Title`;
- the **extracted components**: an append-only list of component copies paired with the paths of
  their sources, written by the `T-Extract-*` tools. Nothing can be removed or reordered;
  extracting the same element twice adds it twice.

The footnote itself is not stored: it is generated from the extracted components when the agent
finishes (see [inputs_outputs.md](inputs_outputs.md#outputs)). `T-Get-Element-Content` only reads
the cited document.

### T-Get-Element-Content: `get_element_content`

Renders a readable element of the cited document as plain text, so that the agent can decide what
to extract.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `path_to_element` | `str` | Path of a `Paragraph`, `Figure` or `MathExpression`. |

**Output**: `str`, depending on the element type:

```
# Paragraph: text as on paper, citations grouped as "[2, 3]", inline math as "$...$"
The two most commonly used attention functions are additive attention [2], and dot-product
(multiplicative) attention. ... except for the scaling factor of $1 / sqrt(d)$.

# Figure: caption and subfigures with their ids
Caption: A diagram illustrating the architecture of the Transformer model.
Subfigures:
	ImgSubfigure (id: subfigure_1): The multi-head attention mechanism.
	TableSubfigure (id: subfigure_2): Hyperparameters of the model.

# MathExpression
Format: LaTeX
Expression: \mathrm{Attention}(Q, K, V) = \mathrm{softmax}(QK^T / \sqrt{d_k}) V
```

**State effect**: none.

**Errors**:
- The element is a `Section` or a `List` -> its content cannot be read; read the elements it
  contains instead (paths from the tree of contents).

### T-Extract-Paragraph-Citation: `extract_paragraph_citation`

Extracts the fragment of a paragraph between two text anchors as a new paragraph. Inline
references and math expressions inside the fragment are kept as components, not as text.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `path_to_paragraph` | `str` | Path of the paragraph. |
| `begin_anchor` | `str` | Text the fragment starts with, copied verbatim from `T-Get-Element-Content`. |
| `end_anchor` | `str` | Text the fragment ends with, copied verbatim. |

Each anchor is located by its longest common substring with the rendered paragraph; at least
`min_citation_match_threshold` of the anchor must match.

**Output**: `None`

**State effect**: appends the fragment paragraph with the paragraph's path.

**Errors**:
- The path is not a paragraph.
- An anchor shares no text with the paragraph, or matches below the threshold -> the message
  lists the matched and unmatched parts of the anchor and the closest text in the paragraph.

### T-Extract-Whole-Paragraph: `extract_whole_paragraph`

Extracts an entire paragraph; preferred over `T-Extract-Paragraph-Citation` when the whole
paragraph is relevant.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `path_to_paragraph` | `str` | Path of the paragraph. |

**Output**: `None`

**State effect**: appends a copy of the paragraph with its path.

**Errors**:
- The path is not a paragraph.

### T-Extract-Figure: `extract_figure`

Extracts a figure with its caption and all or selected subfigures (images or tables).

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `path_to_figure` | `str` | Path of the figure. |
| `subfigures_ids` | `list[str] \| None` | Ids of the subfigures to keep (as shown by `T-Get-Element-Content`); `None` keeps all. |

**Output**: `None`

**State effect**: appends a copy of the figure, keeping its `component_id`, with its path.

**Errors**:
- The path is not a figure.
- Some subfigure ids do not exist -> the message lists the invalid and the existing ids.

### T-Extract-Equation: `extract_equation`

Extracts a standalone (displayed) equation, i.e. a `MathExpression` placed directly in a section.
Inline math is extracted together with its paragraph.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `path_to_equation` | `str` | Path of the math expression. |

**Output**: `None`

**State effect**: appends a copy of the equation, keeping its `component_id`, with its path.

**Errors**:
- The path is not a math expression.

### T-Set-Footnote-Title: `set_footnote_title`

Sets the title of the footnote. Meant to be called once the content is established; the title
describes the footnote's role in explaining the citation (e.g. `Description of additive attention
mechanism`), not the cited paper's title.

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `title` | `str` | The footnote title. |

**Output**: `None`

**State effect**: overwrites the footnote title.

**Errors**: none.
