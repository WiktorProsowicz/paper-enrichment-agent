# <Agent display name>: core

<!-- One or two sentences: what the agent is for. -->

## Agent architecture

<!-- The paradigm in one sentence, e.g. "A ReAct agent: a single reason + act loop over the
tools." For a router + subagents, describe the router's decision. -->

```mermaid
flowchart TD
    start((START)) --> <first_node>
    <first_node> --> <reasoning_node>
    <reasoning_node> -->|tool calls| <action_node>
    <reasoning_node> -->|no tool calls| stop((END))
    <action_node> --> <reasoning_node>
```

| Node | MLflow span | Role | Prompt parts added |
| --- | --- | --- | --- |
| `<node_name>` | `<span name>` | <what the node does> | <templates / messages appended, or -> |

| Template | Message role | Shows to the agent |
| --- | --- | --- |
| `<template name>` | <system / user> | <what it tells the agent and which inputs it presents> |

### Configuration

<!-- Only parameters that change the agent's behavior or what it sees. No defaults, no generic
parameters (recursion limit, temperature, URLs, credentials, timeouts). -->

| Parameter | Influence on the execution |
| --- | --- |
| `<parameter>` | <influence> |

## Important info

<!-- Facts one must know about the agent before designing or implementing it, each starting with
a bold one-sentence claim. No implementation details. -->

- **<Claim>.** <explanation>

## Changelog

<!-- One row per merge to main, newest first. Unmerged changes go to the `unreleased` row. -->

| Date | Merge commit | Changes |
| --- | --- | --- |
| unreleased | - | <change><br><change> |
