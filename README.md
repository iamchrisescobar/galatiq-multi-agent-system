# Invoice Processing Automation

A local multi-agent invoice processing system built for the Galatiq FDE case study. It automates invoice ingestion, deterministic validation, approval reasoning, critique, and final payment/rejection handling.

The system is designed around one principle: **LLMs reason, deterministic code enforces hard business controls.**

## MVP scope

This is a local prototype. External banking and enterprise integrations are mocked, and the inventory system is SQLite-backed. The focus of the MVP is reliable workflow orchestration, safe decision boundaries, error handling, observability, and handling imperfect invoice data.

## Workflow

```text
Invoice
  |
  v
Ingestion Agent
  |
  v
Deterministic Validation
  |
  v
Approval Policy
  |
  v
Approval Agent <------+
  |                   |
  v                   |
Independent Critic ---+  revise if < limit
  |
  | accept
  v
Final Decision
  |
  +------ approve ------> Mock Payment
  |
  +------ reject -------> Rejection Log

If the approval/critic loop cannot converge within the retry limit,
the invoice is written to a manual-review queue instead.
```

### What each stage does

- **Ingestion** uses Grok to extract structured invoice data from TXT, JSON, CSV, XML, and text-based PDF files.
- **Validation** checks required fields and inventory constraints in SQLite. Repeated product lines are aggregated before stock validation, and narrow normalization handles known formatting/OCR variations.
- **Approval** combines deterministic policy with an LLM approval agent. Invoices above $10,000 require additional scrutiny but are not automatically rejected.
- **Critique** independently reviews the proposed decision. It can accept the decision or return revision instructions to the approval agent.
- **Final handling** pays approved invoices, persists rejected invoices with reasoning, or creates a pending manual-review work item when automated reasoning cannot safely converge.

The approval revision loop is bounded to prevent infinite agent cycles.

## Run locally

Requires Python and an xAI API key.

```bash
pip install -r requirements.txt
```

Set your llm provider, model, model configuration and API key in your .env file. Refer to .env.example:

```bash
# Active LLM provider: openai or xai
LLM_PROVIDER=xai

# Use a model available to your provider account.
LLM_MODEL=grok-build-0.1

LLM_TEMPERATURE=0
LLM_MAX_TOKENS=2048
LLM_TIMEOUT_SECONDS=60
LLM_MAX_RETRIES=2

# Provider credentials
OPENAI_API_KEY=KEY_GOES_HERE
XAI_API_KEY=KEY_GOES_HERE
```

Process an invoice:

```bash
python main.py --invoice_path=data/invoices/invoice_1001.txt
```

For a more technical view, including extracted data, policy assessment, agent decisions, critique output, revisions, audit events, and timing use the *--verbose* flag:

```bash
python main.py --invoice_path=data/invoices/invoice_1001.txt --verbose
```

To initialized the database with the seed data:
```bash
python -m invoice_system.database
```

## Example outcomes

The provided invoice set exercises both normal and problematic cases:

| Scenario | Behavior |
| --- | --- |
| Clean invoice within inventory | Approved and mock payment executed |
| Requested quantity exceeds stock | Rejected and logged |
| Unknown / zero-stock product | Rejected and logged |
| Invalid or missing invoice data | Rejected with validation reasoning |
| Repeated product lines | Quantities aggregated before inventory validation |
| OCR / formatting inconsistencies | Narrow normalization before structured extraction |
| Approval and critic cannot converge | Added to `data/manual_reviews.jsonl` |

Final rejections are written to `data/rejections.jsonl`.

## Design decisions

**Deterministic controls over LLM judgment.** Inventory checks, required fields, approval thresholds, routing, retry limits, and payment gating are enforced in code. The LLM cannot override a failed validation result.

**Independent critique before finalization.** Approval reasoning is reviewed by a separate critic. Critic feedback is fed back into the approval agent when revision is required.

**Safe failure behavior.** Agent revision loops are bounded. Unresolved cases become human-review work items instead of being silently approved or rejected. Payment failures are not blindly retried because payment is a side effect.

**Preserve source data, normalize only where necessary.** The extracted invoice keeps the original line items. Validation performs canonical product resolution and cumulative inventory checks without rewriting the source record.

## Testing

Run the full test suite with:

```bash
python -m pytest -v
```

Tests cover ingestion, document loading, deterministic validation, approval policy, agent/critic guardrails, revision limits, manual review, payment/rejection routing, persistence, failures, and CLI output.
