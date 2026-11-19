"""Scorers shared by the evaluation suites of the MCP-based ReAct agents.

The agents of the app emit the same kinds of spans, regardless of their task: a reasoning step
(``agent_reasoning``), an action step (``agent_action``) and a tool call (``call_tool <name>``,
emitted by :mod:`paper_enrichment_agent.common.mcp_agent_utils`). The scorers in this module
measure the agents' execution based on these spans. Keep the span names in sync with the agents.
"""

import re

from mlflow.entities import Trace
from mlflow.entities.assessment import Feedback
from mlflow.genai import Scorer, scorer

CALL_TOOL_SPAN_PATTERN = re.compile(r'^call_tool\b')
AGENT_REASONING_SPAN_NAME = 'agent_reasoning'
AGENT_ACTION_SPAN_NAME = 'agent_action'


@scorer
def agent_step_count_metrics(trace: Trace) -> list[Feedback]:
    """Counts the tool calls and the reasoning and action steps of the agent."""

    return [
        Feedback(
            name='call_tool_count', value=len(trace.search_spans(name=CALL_TOOL_SPAN_PATTERN))
        ),
        Feedback(
            name='agent_reasoning_count',
            value=len(trace.search_spans(name=AGENT_REASONING_SPAN_NAME)),
        ),
        Feedback(
            name='agent_action_count', value=len(trace.search_spans(name=AGENT_ACTION_SPAN_NAME))
        ),
    ]


def spawn_specific_tool_usage_metrics(tool_name: str) -> Scorer:
    """Spawns a metric for tracking the usage of a specific tool."""

    def specific_tool_usage_metrics(trace: Trace) -> list[Feedback]:
        """Measures the call count and the success rate for a specific tool.

        The success rate is not reported if the tool was not called, so that the run is not
        scored on (and validated against) a behaviour that did not occur.
        """
        call_tool_spans = [
            span
            for span in trace.search_spans(name=CALL_TOOL_SPAN_PATTERN)
            if span.attributes['tool_name'] == tool_name
        ]

        successful_tool_calls = [
            span for span in call_tool_spans if not span.attributes['tool_is_error']
        ]

        call_count_feedback = Feedback(name=f'call_count:{tool_name}', value=len(call_tool_spans))

        if not call_tool_spans:
            return [call_count_feedback]

        return [
            call_count_feedback,
            Feedback(
                name=f'success_rate:{tool_name}',
                value=len(successful_tool_calls) / len(call_tool_spans),
            ),
        ]

    return scorer(specific_tool_usage_metrics)
