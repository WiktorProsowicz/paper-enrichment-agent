# paper-enrichment-agent

Paper Enrichment Agent is a tool designed to convert arXiv survey papers into full-fledged textbooks. Consider the following problem:

You want to read up on a chosen topic, such as "Paragraph Text-to-Speech synthesis", and you find a relevant survey paper on arXiv. However, most paragraphs take the form:

```
... Multiple works attempt to address the problem of lack of high quality paragraph data [1, 2, 3, 4]. In [5], the authors use context-extrapolation and generative context-enrichment techniques to train on single-sentence data. The authors of [6] build a single-sentence model, conditioned on GST-based [7] and BERT-based [8] embeddings, that leverages a mixture attention mask.
```

You realize that you are unlikely to grasp the topic fully without additional context, background information, and examples. This is where Paper Enrichment Agent comes in:

- The paper is parsed and converted into ebook format that is more reader-friendly and easier to navigate using a kindle-like device
- Each reference to a paper is automatically expanded into a context-aware citation that provides relevant quotes, figures and tables from the cited paper
- Each processed reference paper is effectively converted into a set of footnotes, which explain the context of the citation and provide additional background information

## Changelog

The changes are provided in the [CHANGELOG.md](CHANGELOG.md) file.