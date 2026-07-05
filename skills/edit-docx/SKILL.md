---
name: edit-docx
description: >-
  Apply a set of edits to a Microsoft Word .docx file and produce a new downloadable
  .docx that preserves the original's formatting. Use when a user wants to revise, update,
  or amend an existing Word document and get back a single merged .docx. Works on .docx
  only — not PDFs, plain text, or other formats. The edits can come from a separate change
  document (an amendment, revision memo, redline, or change request) or from instructions
  given directly in the request. Needs the base .docx to already be uploaded; a change
  document, if used, must be uploaded too.
metadata:
  requires:
    bins: []
  install:
    - type: pip
      packages:
        - python-docx
---

# edit-docx

Apply a set of changes to a base `.docx` and produce a merged `.docx`. You (the agent)
drive the merge — deriving the edits, applying them with the bundled helper, and
returning a download link. The logic lives here in the skill, not in the platform.

Scope: **Microsoft Word `.docx` files only.** Both the base and the merged output are
`.docx`; other formats (PDF, plain text, Markdown) are not supported. The *domain* is
open, though — contract-and-amendment is the canonical case, but the same mechanism
handles a policy plus a revision memo, a spec plus a change request, a report plus
reviewer comments, or any `.docx` plus a set of changes.

## Inputs

Every request has a **base document** and a **change source**.

- **base** — the document to edit (`base_doc_id`), an already-uploaded `.docx`.
- **change source** — where the edits come from, one of:
  - a **change document** — a second uploaded document that describes the changes
    (`change_doc_id`): an amendment, revision memo, addendum, or change request.
  - **inline instructions** — the changes stated directly in the user's request
    ("in the base contract, change the payment term to 45 days and add a Delaware
    governing-law clause").

If the base id is unclear, or a change is described but you can't tell whether it points
to an uploaded document or is meant inline, ask before proceeding.

## Workflow

1. **Understand the changes.** Read the source of the edits:

   - If there's a **change document**, use `read_document` on it (and on the base as
     needed) to read the text. Changes are typically written as operations — "Section
     4.2 is deleted and replaced with…", "a new Section 9 is added…".
   - If the changes are given **inline**, use `read_document` on the base as needed to
     locate the target text, and take the edits from the request.

   Turn the changes into a list of edit operations, each one of:

   - `replace` — swap a paragraph's text. `anchor_text` = a short verbatim snippet
     from the **base** that uniquely locates the target paragraph; `new_text` = the
     replacement.
   - `delete` — remove a paragraph. `anchor_text` locates it.
   - `insert_after` — add a new paragraph after an existing one. `anchor_text` locates
     the predecessor; `new_text` is the added text.
   - `append` — add a paragraph at the end. `new_text` only.

   `anchor_text` must be copied verbatim from the base so it can be matched. Preserve
   the intent of the changes exactly; never invent content.

2. **Fetch the base file.** Call `fetch_document` with the `base_doc_id` to materialize
   the raw `.docx` to a local path (`read_document` only returns text — you need the
   binary to preserve formatting).

3. **Apply the operations.** Write the operations to an `ops.json` file shaped as
   `{"operations": [ ... ]}`, then run the bundled helper with the `shell` tool
   (substitute the skill base directory printed above):

   ```
   python <skill base directory>/apply_operations.py \
     --original <fetched path> --ops ops.json --output merged.docx
   ```

   It edits the base at the run level (preserving fonts/styles/numbering) and prints a
   JSON report: a `matched` flag per operation and an `unmatched` count.

4. **Check the report.** If any operation is `unmatched`, its anchor wasn't found —
   usually the change referenced text the base doesn't contain. Do not silently drop
   these; call them out to the user.

5. **Save and return.** Call `save_artifact` with the path to `merged.docx` and a
   descriptive `filename` (e.g. `<document-name>-revised.docx`). It returns a
   ready-made markdown download link — include that exact link in your answer, along
   with a short summary of what changed and any unmatched operations. (The link is
   also appended to your answer automatically, so the user always gets a download.)

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
</content>
