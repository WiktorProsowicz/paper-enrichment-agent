# <Agent display name>: inputs and outputs

## Inputs

### Specification

<!-- Structured value: the model name and a JSON listing of its shape, values as
"<type: description>" placeholders. Free-form text: a description of what it contains. Several
values: a list or a table. -->

Model: `<Model>`

```json
{
  "<field>": "<type: description>"
}
```

| Prompt variable | Source |
| --- | --- |
| `<variable>` | <input field, or how it is derived from the inputs, in words> |

### Examples

#### IN-<Name>

<!-- When this class of input occurs, then a concrete, shortened example. -->

```json
{}
```

## Outputs

### Specification

<!-- What the agent answers. If the agent also produces a result through its tools' state,
describe it here as the other kind of output (no example classes for it). -->

Model: `<Model>`

```json
{
  "<field>": "<type: description>"
}
```

### Examples

#### OUT-<Name>

<!-- When this class of output is produced, then a concrete, shortened example. Errors and
degenerate results are not classes. -->

```json
{}
```
