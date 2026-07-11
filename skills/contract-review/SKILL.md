---
name: contract-review
description: >-
  Review a Microsoft Word .docx contract from one party's perspective and produce a
  structured, clause-by-clause analysis a reviewer can act on: for each clause, a risk
  level with rationale, a suggested change, and any cross-clause contradictions. First
  confirms which party the review represents and the review posture (dominant, neutral,
  or disadvantaged), then analyzes every clause through that lens and writes a review
  file (ops.json) of suggestions the user can accept, reject, or refine — and, on request,
  a tracked-changes redline .docx of the accepted changes. Use when a user uploads a
  contract and wants it reviewed, risk-assessed, or marked up. Works on .docx only.
  Needs the contract already uploaded.
metadata:
  requires:
    bins: []
  install:
    - type: pip
      packages:
        - python-docx
---

# contract-review

Review an uploaded `.docx` contract **from one party's side** and produce a structured,
clause-by-clause assessment: per clause a **risk** (level + rationale), an optional
**suggested change**, and any **contradictions** with other clauses. The suggestions are
written to a **review file** (`ops.json`) the user can accept / reject / refine over the
conversation; accepted changes become a **tracked-changes redline `.docx`** via the
`edit-docx` skill.

You (the agent) do the judgment — deciding risk and drafting suggested language for the
represented party. Two bundled helpers do the mechanical, error-prone parts so your
output stays reliable:

- `segment_clauses.py` — splits the base `.docx` into clauses and gives every paragraph
  a stable id and its **verbatim** text. Anchor your suggestions to these paragraphs by
  id; never hand-copy anchor text.
- `build_ops.py` — turns your analysis into the review file (resolving those ids to
  matchable anchors) and later projects the review file into `edit-docx` operations.

Scope: **Microsoft Word `.docx` only.** This skill *analyzes and marks up*; it does not
answer general questions about the contract (use ordinary retrieval for that).

## Inputs

- **base** — the contract to review (`base_doc_id`), an already-uploaded `.docx`.
- **represented party** — which party the review speaks for. Derive the candidate parties
  from the contract and **confirm with the user** before analyzing.
- **review position** — one of `dominant`, `neutral`, `disadvantaged`. This sets how
  aggressive the suggested changes are (a `dominant` party pushes harder terms; a
  `disadvantaged` one seeks protections). **Confirm with the user.**

If the base id is unclear, ask before proceeding.

## Workflow

1. **Fetch the contract.** Call `fetch_document` with `base_doc_id` to materialize the raw
   `.docx` to a local path (you need the binary, not just extracted text).

2. **Identify parties and confirm the lens.** Use `read_document` on the base to find the
   contracting parties, the contract type, and the governing law. Then **ask the user**
   which party the review represents and the review position (`dominant` / `neutral` /
   `disadvantaged`). Wait for the answer — do not analyze until both are confirmed. This
   confirmation is the point of the skill; the whole analysis is written through that lens.

3. **Segment the clauses.** Run the segmenter with the `shell` tool (substitute the skill
   base directory printed above):

   ```
   python <skill base directory>/segment_clauses.py \
     --original <fetched path> --output clauses.json
   ```

   `clauses.json` is `{"clauses": [{"clause_id", "heading", "paragraphs": [{"para_id",
   "text"}]}]}`. Read it back.

4. **Analyze each clause.** For every clause, judge it *for the represented party at the
   chosen position* and write a raw analysis file `analysis.json`:

   ```json
   {
     "base_doc_id": "<base_doc_id>",
     "meta": {"parties": ["Provider", "Client"], "representative_party": "Client",
              "review_position": "disadvantaged", "contract_type": "MSA",
              "governing_law": "New York"},
     "analyses": [
       {"clause_id": "c1",
        "risk": {"level": "high", "rationale": "30-day payment term strains the Client."},
        "contradicts": [],
        "suggestion": {"op": "replace", "para_id": "c1.p1",
                       "new_text": "Payment shall be due within 45 days of invoice receipt."}}
     ]
   }
   ```

   Rules:
   - `risk.level` is one of `high` / `medium` / `low` / `none`.
   - `suggestion` is `null` when the clause needs no change. Otherwise `op` is one of
     `replace`, `delete`, `insert_after`, `append` (same vocabulary as `edit-docx`):
     `replace`/`delete`/`insert_after` need a `para_id` naming the target paragraph from
     `clauses.json`; `replace`/`insert_after`/`append` need `new_text`. **Reference
     paragraphs by `para_id`** — do not write `anchor_text` yourself.
   - `contradicts` lists other `clause_id`s this clause conflicts with (e.g. a payment
     term that disagrees with a fees schedule). Include clauses you'd flag even when you
     don't suggest a change.
   - Draft `new_text` in the represented party's favor, proportionate to the position.
     Never invent obligations the parties didn't discuss; keep clause intent intact.

5. **Build the review file.** Resolve ids to anchors and default every verdict to
   `pending`:

   ```
   python <skill base directory>/build_ops.py finalize \
     --analysis analysis.json --clauses clauses.json --output review.json
   ```

   A non-existent `para_id` or a malformed op fails loudly here — fix the analysis and
   rerun rather than shipping a review that can't be applied.

6. **Persist the review as working state.** Call `save_artifact` with `review.json` and a
   descriptive `filename` (e.g. `<contract-name>-review.json`). Note the returned
   **artifact id** — it is the handle for the rest of the conversation. Reopen it later
   with `fetch_artifact`, patch verdicts or suggestions, and `save_artifact` a fresh copy.

7. **Summarize for the user.** Report the clauses by risk (highest first), the suggested
   changes, and any contradictions, and tell the user they can accept, reject, or refine
   each suggestion — and ask for a redline when ready.

## Producing the redline

When the user wants to see or apply changes, project the review file to `edit-docx`
operations and run the `edit-docx` apply helper:

```
# Preview (all suggestions) — use --all; drop it to redline only accepted verdicts.
python <skill base directory>/build_ops.py to-edit-ops \
  --review review.json --all --output edit_ops.json

python <edit-docx base directory>/apply_operations.py \
  --original <fetched path> --ops edit_ops.json --output redline.docx --author "contract-review"
```

`to-edit-ops` without `--all` includes only clauses whose `verdict` is `accepted`, so the
final apply reflects exactly what the user approved. The anchors were taken verbatim from
the base during segmentation, so they match; still check `apply_operations`' `unmatched`
count and surface any misses. `save_artifact` the `redline.docx` and include the returned
download link in your answer, noting it is a tracked-changes redline reviewable in Word.

## Refining across turns

The review file is the single source of truth for the review. When the user accepts,
rejects, or asks to reword suggestions ("accept clause 1, reject clause 3, soften clause
5"), don't hand-edit the JSON — apply the changes deterministically with `build_ops.py
patch` so a mistyped `clause_id` or verdict fails loudly instead of corrupting the file:

1. `fetch_artifact` the current review file to a local path.
2. Apply the changes:

   ```
   # Verdicts go on the command line; rewording/dropping a suggestion goes in a patch file.
   python <skill base directory>/build_ops.py patch \
     --review review.json --accept c1 --reject c3 \
     --patch changes.json --output review.json
   ```

   `changes.json` is optional and only needed to reword or drop a suggestion:

   ```json
   {"suggestions": {
     "c5": {"new_text": "Payment shall be due within 30 days of invoice receipt."},
     "c8": null
   }}
   ```

   A `null` drops the clause's suggestion so it's never applied; a fields dict is merged
   into the existing suggestion (keeping the baked `anchor_text`, so a reworded change
   still matches the base). Verdicts may also be given in the patch file under `"verdicts"`
   instead of the flags.

3. `save_artifact` the updated review file and regenerate the redline (see above).

When the review is complete, `delete_artifact` the review file (and any stale redline
previews) to clean up the working state.

## Helper reference

- `segment_clauses.py --original in.docx [--output clauses.json]` — emit clauses with
  per-paragraph verbatim anchors.
- `build_ops.py finalize --analysis a.json --clauses c.json [--output review.json]` —
  resolve `para_id`s to anchors, default verdicts to `pending`, validate.
- `build_ops.py patch --review review.json [--accept ID...] [--reject ID...]
  [--pending ID...] [--patch changes.json] [--output review.json]` — apply
  accept/reject/refine decisions; validates every `clause_id`, verdict, and resulting op.
- `build_ops.py to-edit-ops --review review.json [--all] [--output edit_ops.json]` —
  project to `edit-docx` operations (accepted-only unless `--all`).

## Limitations

Segmentation splits on headings and section numbers (`ARTICLE`/`SECTION`, `4`, `4.2`,
roman numerals) and ALL-CAPS title lines; unusual numbering may under-segment, though
anchors still resolve at the paragraph level. Redlining inherits `edit-docx`'s scope —
body paragraphs only, not tables/headers/footers, and a `replace` swaps a whole paragraph
rather than a word-level diff. The analysis is decision support, not legal advice.
