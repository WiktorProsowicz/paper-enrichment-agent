---
name: agent-spec
description: Write or update the specification of an LLM agent in docs/agents_spec/<agent_name>/ (core, inputs/outputs, tools, procedures). Use before implementing a new agent or a change of an agent's graph, prompts, tools, inputs or outputs, or when the user asks to document or specify an agent.
---

# Agent specification

An agent specification describes **what an agent does and how it behaves**, independently of its
implementation. It is written or updated **before** the agent is implemented or changed, and the
implementation follows it. A reviewer must be able to check the implementation and the LLM
evaluation against it.

Because of that, a specification never refers to places in the code: no module paths, file links,
class methods, service routes or names of the callers. Names that belong to the agent's contract
are fine: the names of the input/output models (`FootnoteEnrichmentRequest`), of the tools, of
the graph nodes, of the prompt templates and of the configuration parameters.

The specification lives in `docs/agents_spec/<agent_name>/`, where `<agent_name>` is the
snake-case name of the agent, e.g. `footnote_enrichment_agent`, `paper_access_agent`.

```
docs/agents_spec/<agent_name>/
├── core.md             # Architecture (graph, prompts, configuration), important info, changelog
├── inputs_outputs.md   # Inputs and outputs of the agent, with example classes
├── tools.md            # Tools grouped by shared state, each with a T-* code
└── procedures.md       # Primary execution paths as numbered steps + activity diagrams
```

Templates of all four files are in [templates/](templates/). Copy them and replace every
`<placeholder>`; delete the template comments (`<!-- ... -->`) from the final files.

## Workflow

### 1. Gather the facts

- **New agent**: take the design from the user: the task, the inputs and outputs, the tools and
  their state, the strategy. Ask about anything the four files need and the user did not give
  (e.g. the output classes or the state shared by the tools); do not invent behavior.
- **Changed agent**: read the existing specification, then the agent's current implementation
  (graph, prompts, tools, the configuration) and its LLM eval suite, to know the current behavior.
  Then take the intended change from the user.
- **Agent implemented before its specification**: derive the specification from the
  implementation and the prompts. Report every inconsistency you find (prompt vs tool
  descriptions, prompt vs code) to the user instead of writing it into the specification.

### 2. Write or update the four files

Follow the format rules below and the templates.

- **New agent**: create the directory from the templates.
- **Changed agent**: edit only the affected sections. Keep the tool and procedure codes stable: a
  renamed tool keeps its code unless its purpose changes; a removed tool's code is not reused.
- **Changelog**: one row per **deployment**, i.e. per merge to `main`. Changes not merged yet go
  to a single `unreleased` row at the top, which collects all of them until the merge; on the
  merge, fill in its date and merge commit.

### 3. Cross-check

- Every tool of the agent appears in `tools.md` under the name the agent sees, and no other tool
  does. Every tool code used in `procedures.md` is defined in `tools.md`.
- Every node and edge of the execution graph appears in the diagram of `core.md`, and every
  prompt template in the templates table.
- Every output class code used in `procedures.md` is defined in `inputs_outputs.md`.
- The examples in `inputs_outputs.md` match the specified shapes (field names, types, required
  fields).
- Mermaid blocks parse: wrap node labels containing parentheses, brackets, slashes or colons in
  `"..."`.

### 4. Update the surroundings

If the directory is new, add it to the `## Project structure` tree of `README.md`.

## Format rules

### General

- GitHub-flavored Markdown, one `#` title per file: `# <Agent display name>: <file topic>`.
- Wrap lines at 100 characters, except tables, diagrams and JSON listings.
- Diagrams are Mermaid (` ```mermaid `). Use `flowchart TD`; there is no native activity diagram
  type, so activity diagrams use the conventions in *procedures.md* below.

### Codes

| Entity | Format | Example |
| --- | --- | --- |
| Tool | `T-<Verb>-<Object>` in Title-Kebab-Case, derived from the tool name | `T-Extract-Figure` |
| Procedure | `P-<Name>` in Title-Kebab-Case | `P-Compose-Footnote` |
| Input / output example class | `IN-<Name>` / `OUT-<Name>` | `OUT-Arxiv-Source` |

Codes are unique within an agent's specification and referenced verbatim in the other files.

### core.md

Exactly three `##` sections:

1. **Agent architecture**:
   - The paradigm in one sentence (ReAct, router + subagents, plan-and-execute, ...).
   - A Mermaid diagram of the execution graph with the node names. For a ReAct agent it is the
     reason/act loop.
   - A nodes table `| Node | MLflow span | Role | Prompt parts added |`. The last column names
     the prompt templates (or other messages, e.g. tool results) the node appends to the
     conversation, `-` if none.
   - Right below it, a prompt templates table `| Template | Message role | Shows to the agent |`,
     one row per template, saying what the template tells the agent and which inputs it presents.
   - A `### Configuration` subsection with a table `| Parameter | Influence on the execution |`.
     List only parameters that change how the agent behaves or what it sees (e.g. the size of
     the citation context, the number of search results shown, matching thresholds of the
     tools). Leave out generic ones (recursion limit, temperature, URLs, credentials, timeouts)
     and do not list default values.
2. **Important info**: a bullet list of facts one must know about the agent **before** designing
   or implementing it, which do not follow directly from the other files. Each bullet starts with
   a bold one-sentence claim. Good: "The footnote is the tools' state, not the LLM's answer",
   "Extractions cannot be undone", "Each resolution is independent". Not here: implementation
   details (how the tools are served, in-process endpoints, checkpointers, span coupling) and
   inconsistencies of the current code (report those to the user).
3. **Changelog**: a table `| Date | Merge commit | Changes |`, newest first, one row per merge to
   `main` (see the workflow). Dates are `YYYY-MM-DD`; the `unreleased` row has `-` as the merge
   commit. List the changes of a row as short items separated by `<br>`.

### inputs_outputs.md

Exactly two `##` sections, **Inputs** and **Outputs**. Each has two `###` subsections:
- **Specification**: for a structured value, the name of its model and a ` ```json ` listing of
  its shape with `"<type: description>"` placeholders as values; for free-form text, a
  description of what it must contain; for several values, a list (or a table) with one entry
  per value. For inputs, add a table `| Prompt variable | Source |` mapping the prompt variables
  to the inputs, including values derived from them (e.g. a rendered table of contents),
  described in words, not with code.
- **Examples**: one `####` subsection per *class* of input or output: a qualitatively different
  kind of valid value the agent must handle or produce (e.g. "arXiv source", "PDF source",
  "not found" for a resolution agent; "detailed citation" vs "group citation" for an input).
  Each class has a code, one sentence on when it occurs, and a concrete, shortened example.
  Errors, failures and degenerate results (limits hit, empty results) are **not** classes.

The output is what the agent itself answers (its final message, or the decoded value of it). If
the agent also produces a result through its tools' state (e.g. a document built by the tools),
describe it in the Outputs specification as the other kind of output, without example classes,
and mention it in *Important info* of `core.md`.

### tools.md

- No description of how the tools are exposed (MCP servers, transports, endpoints). The file may
  start with the facts common to all tools as the agent sees them, e.g. the addressing scheme of
  elements or that a tool error is returned to the agent, which can correct the call.
- One `##` section per **tool group**: tools that share state. Start the section with the state:
  what it holds, when it is created and discarded, which tools read and which modify it. Tools
  that share no state each get their own `##` section with *State: none*.
- One `###` subsection per tool, titled `` <code>: `<tool name>` ``, containing:
  - a short description of what the tool does (not a copy of its description for the LLM);
  - **Input**: a table `| Argument | Type | Description |`;
  - **Output**: the type and a short example, or `None` for state-modifying tools;
  - **State effect** (in groups with state): what is read or modified;
  - **Errors**: the conditions under which the tool returns an error to the agent.

### procedures.md

- The first `##` section, **Primary execution paths**, has one `###` subsection per path, titled
  `<code>: <name>`. A path is a qualitatively different way an invocation can run: a subagent, a
  branch of a router, or, for a single-loop agent, the strategy the prompt prescribes.
- Each procedure has a one-sentence goal, a trigger (when this path runs), and a numbered list of
  steps starting from **1** in every procedure. A step says what is done, which tools (by code)
  are called, and its conditional transitions written as `if <condition> -> go to step N`,
  `else -> continue` or `-> finish with <output class code>`. The **last step is the agent's
  final answer**; post-processing of the answer outside the agent is not a step.
- Below the steps, an activity diagram in Mermaid:
  - start: `start((start))`, end: `stop(((end)))`, one end node per output class if they differ;
  - actions: rectangles labeled `N. <short action>` with the step number;
  - decisions: rhombi `{"<question>?"}` with labeled edges `-->|yes|` / `-->|no|`;
  - loops are drawn as edges back to the earlier action.
- A second `##` section, **Failure handling**, covers only **tool errors**: which errors the agent
  is expected to recover from and how (e.g. correcting an anchor or a path, falling back to the
  next source). No other sections: limits, decoding failures and the prompt's guidelines are not
  part of this file.
