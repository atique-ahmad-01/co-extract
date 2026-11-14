# Change order extraction

Give it a change order (PDF or pasted text) and it returns JSON with each field's value, a
confidence score and the quote it came from. If a field isn't in the document, it's `null`.
If something looks wrong, `needs_review` is `true`.

The JSON Schema is in `schema/change_order.schema.json`, and the outputs for the sample
documents are in `outputs/`.

## Setup

```bash
make setup
cp .env.example .env
```

## Run with OpenAI

Set these in `.env`:

```
LLM_PROVIDER=openai
LLM_API_KEY=your-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Then:

```bash
make extract FILE=path/to/change_order.pdf
```

## Run with Claude

Set these in `.env`:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key
ANTHROPIC_MODEL=claude-opus-5-5
```

Then:

```bash
make extract FILE=path/to/change_order.pdf
```

## Run the web UI

```bash
make app
```

Open http://localhost:8501, choose OpenAI or Claude, and upload a PDF or paste text.

## Tests

```bash
make test
```

## Approach

1. Read the text from the PDF or the pasted input.
2. Ask the LLM for each field, along with the exact text it took the value from.
3. Run simple regex patterns on the same text as a second check.
4. Clean up values: dates as YYYY-MM-DD, credits as negative amounts, schedule impact in days.
5. Check the quoted text really exists in the document. If it doesn't, the value becomes `null`.
6. Check business rules: required fields, sensible ranges, the amount's sign matches words like
   "credit" or "increase", and the old contract sum plus this change equals the new sum.

## Why the confidence scores mean something

The score doesn't come from the model rating itself. It starts at 1.0 and drops when a check fails:

- the quote is only an approximate match (×0.85)
- the value can't be read back from its quote (×0.6)
- the regex check finds nothing (×0.9) or a different value (×0.7)
- repeated LLM runs disagree (down to ×0.5)
- a business rule fails (×0.5)

Anything under 0.6 goes to review. I tested this on six hand-labelled documents: with Claude,
96% of fields were correct. Of the two wrong fields, one scored low and was sent to review, and
one ("Owner (via Bulletin 6)" instead of "Owner") got through at 0.70. Six documents is a small
test, so the next step would be checking it against real change orders.

## Failure modes

- **Scanned PDFs** with no text come back empty and flagged for review.
- **OCR mistakes** like `8,4l0.00` can be silently fixed by the LLM. The quote check catches
  this and sends it to review.
- **Several amounts** in one document is the biggest risk. The contract-sum check catches a
  wrong pick, but only when the document lists the sums.
- **Dates like 03/04/2025** are read as US format (March 4).
- **Approval** is read from signatures; a blank owner signature means pending.
- **"Requested by"** can mean the owner who asked for the change or the contractor who sent it.
- **If the LLM call fails**, the regex results are used and the document is flagged for review.
