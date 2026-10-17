# Judge prompt

Fixture: same layout as data/monitor/judge_prompt.md.

## System

You are a careful annotator.

{rubric}

## User

Score this trace.

<trace>
{trace}
</trace>

Return JSON only: {"level": 0|1|2, "evidence": "<exact substring or empty>", "flag_unreadable": true|false}
