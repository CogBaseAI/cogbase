---
name: merge-contract
description: >-
  Merge a contract amendment into its original, producing a new downloadable .docx that
  preserves the original's formatting. Use when a user wants to apply an amendment,
  addendum, or set of changes to an existing contract and get back a single merged Word
  document. Needs the original and the amendment to already be uploaded documents.
metadata:
  requires:
    bins: []
  install:
    - type: pip
      packages:
        - python-docx
---

# merge-contract

Apply a contract amendment to its original and produce a merged `.docx`. You (the
agent) drive the merge — deriving the edits, applying them with the bundled helper,
and returning a download link. The logic lives here in the skill, not in the platform.

## Inputs

The user's request identifies two already-uploaded documents:

- **original** — the base contract (`original_doc_id`).
- **amendment** — the document describing the changes (`amendment_doc_id`).

If either id is unclear, ask before proceeding.

## Workflow

1. **Understand the amendment.** Use `read_document` on the amendment (and the
   original as needed) to read their text. Amendments are written as operations —
   "Section 4.2 is deleted and replaced with…", "a new Section 9 is added…". Turn
   them into a list of edit operations, each one of:

   - `replace` — swap a clause's text. `anchor_text` = a short verbatim snippet
     from the **original** that uniquely locates the target paragraph; `new_text`
     = the replacement.
   - `delete` — remove a clause. `anchor_text` locates it.
   - `insert_after` — add a new clause after an existing one. `anchor_text` locates
     the predecessor; `new_text` is the added clause.
   - `append` — add a clause at the end. `new_text` only.

   `anchor_text` must be copied verbatim from the original so it can be matched.
   Preserve the amendment's intent exactly; never invent clauses.

2. **Fetch the original file.** Call `fetch_document` with the `original_doc_id` to
   materialize the raw `.docx` to a local path (`read_document` only returns text —
   you need the binary to preserve formatting).

3. **Apply the operations.** Write the operations to an `ops.json` file shaped as
   `{"operations": [ ... ]}`, then run the bundled helper with the `shell` tool
   (substitute the skill base directory printed above):

   ```
   python <skill base directory>/apply_operations.py \
     --original <fetched path> --ops ops.json --output merged.docx
   ```

   It edits the original at the run level (preserving fonts/styles/numbering) and
   prints a JSON report: a `matched` flag per operation and an `unmatched` count.

4. **Check the report.** If any operation is `unmatched`, its anchor wasn't found —
   usually the amendment referenced a section the original doesn't contain. Do not
   silently drop these; call them out to the user.

5. **Save and return.** Call `save_artifact` with the path to `merged.docx` and a
   descriptive `filename` (e.g. `<contract-name>-merged.docx`). Report the download
   path it returns, plus a short summary of what changed and any unmatched operations.

## `apply_operations.py`

```
python apply_operations.py --original in.docx --ops ops.json --output out.docx
```

`ops.json`:

```json
{"operations": [
  {"op": "replace",      "anchor_text": "Payment shall be due within 30 days", "new_text": "Payment shall be due within 45 days."},
  {"op": "delete",       "anchor_text": "Either party may terminate with 60 days notice"},
  {"op": "insert_after", "anchor_text": "Section 8 Term",                       "new_text": "Section 9 Governing Law: State of Delaware."},
  {"op": "append",                                                              "new_text": "Signed as of the amendment date."}
]}
```

## Limitations / upgrade path

The helper walks paragraphs and does clean (non-tracked) edits — it does not touch
tables, headers, or footers, and produces a finished document rather than a redline.
For a lawyer-reviewable **redline** (Word tracked changes), edit the unpacked OOXML
directly (`<w:ins>`/`<w:del>`) instead of using this helper — the operation set you
derive in step 1 is the same; only the apply step changes.
