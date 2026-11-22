# <Agent display name>: tools

<!-- Optional: facts common to all tools as the agent sees them (addressing scheme, how errors
reach the agent). Nothing about how the tools are served. -->

## <Tool group name>

<!-- For a group of tools sharing state: what the state holds, when it is created / discarded,
which tools read it and which modify it. For a standalone tool: "State: none." -->

State: <description>

### T-<Verb>-<Object>: `<tool name>`

<What the tool does, in one to three sentences.>

**Input**

| Argument | Type | Description |
| --- | --- | --- |
| `<arg>` | `<type>` | <description> |

**Output**: `<type>`

```
<short example output>
```

**State effect**: <what is read / modified, or "none">

**Errors**:
- <condition> -> <error message gist>
