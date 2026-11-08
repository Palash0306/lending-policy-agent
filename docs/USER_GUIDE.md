# User guide for testers

This tool reads your bank's lending policy documents and finds the **minimum and maximum limits** and **date cutoffs** for the fields you name. It shows the exact sentence and page each value came from, and it checks every value against the document text. Nothing leaves your computer.

It finds *candidates* for your test design. **Always check the quoted page before you use a value.**

## 1. Open it
Start the server (see `docs/DEPLOYMENT.md`), then open **http://127.0.0.1:8000** in your browser. The line at the top should say `Ollama: ready`.

## 2. Add your documents
In the **Documents** box choose one or more `.pdf` or `.docx` files and press **Upload and index**. Big files take a few minutes. Each file shows how many pieces ("chunks") it was split into.
- Uploading a file with the same name **replaces** the old one. If the new file is broken, your old version stays.
- **remove** deletes a document from the index.
- Scanned PDFs (pictures of pages) cannot be read. You will see "no readable text".

## 3. Ask one question
Type a field name (`loan_tenure`) or a question ("What is the maximum loan amount?") and press **Ask**. One field takes about **4 to 5 minutes** on a laptop. The page shows a timer, so you can tell it is working.
- Ask a follow-up in the same conversation: "what about for commercial loans?". The tool rewrites it into a full question; the line "Searched for" under the answer shows what it really used.
- **Search in** limits the search to one document. **Look for** limits it to numeric limits or date cutoffs. **Quick mode** reads fewer paragraphs: faster, but it can miss a rule.

## 4. Ask many fields at once (Batch)
1. Open the **Batch** tab.
2. Type or paste your fields, one per line (commas also work). Bullets and numbers are ignored and repeats are removed. The counter shows how many fields you have. You can also load a `.txt` or `.csv` file of names; it is read in your browser and not uploaded.
3. Press **Run batch**, then leave it. Each field takes about 4 to 5 minutes, so **20 fields is roughly 1.5 hours**. The page shows progress and a rough time left. Every field is saved as it finishes, so closing the page, a freeze or a restart does not lose finished fields.
4. **Stop after current field** ends the batch early; finished fields are kept.
5. When it finishes press **Download Excel (all fields)**. You get one file with every field.

## 5. Reading an answer
| Label | Meaning |
|---|---|
| **VERIFIED** | The tool found a limit, and the quote, the number or date were all found in the document text. |
| **NEEDS REVIEW** | A person must look before using it: a check failed, two values disagree, only a month was given, or only an example or guidance was found. |
| **NOT FOUND** | No limit was found in the paragraphs read. This does **not** prove the policy has none. |
| **ERROR** | The tool could not answer (for example Ollama is not running). Nothing was guessed. |

Other things you may see:
- **Firm rule / example / guidance**: a *firm rule* is required by the document ("shall", "must"). An *example* is only an illustration ("for example", "some banks"). Do not use an example as a hard limit.
- **Tiers**: several values for one field (72 months below 90 percent LTV, 60 above). There is then no single headline value; all are listed.
- **Report (.md)** saves the answer you are looking at as a text report.
- **Test values (.xlsx)** exports that one answer. **Export all answers (.xlsx)** (top of the conversation) exports the whole conversation.

## 6. The Excel file, sheet by sheet
**Summary**: one row per field.

| Column | Meaning |
|---|---|
| Field | The field or question you asked. |
| Outcome | Verified, Needs review, Not found or Error. |
| Headline limit(s) | The main limit, such as "maximum 30 years". Empty or "several values" when there are tiers or conflicts. |
| Source file, Page | Where the headline came from. |
| Classification | firm_rule, example or guidance. |
| Grounded | Yes if every check passed. |
| Needs review | Yes if a person should look first. |
| Seconds | How long the field took. |

**Test values**: the boundary values to try.

| Column | Meaning |
|---|---|
| Field, Rule | The field and the rule being tested, such as "maximum 72 months (applies to: LTV below 90 percent)". |
| Case | Which point: Min - 1, Min, Min + 1, Max - 1, Max, Max + 1, or Day before cutoff, Cutoff day, Day after cutoff. A field with both a minimum and a maximum gets one six-point set. |
| Test value, Unit | The value to enter. Units are copied from the document and never converted. |
| Expected result | **Accept**, **Reject**, **Confirm with policy owner**, **Rule applies**, **Rule does not apply**, **Confirm (document gives only the month)**, or **Not a firm rule**. |
| Basis | Why the limit value itself is accepted or rejected: *inclusive* ("up to", "at least"), *exclusive* ("less than", "above") or *unclear*. |
| Source file, Page, Quote | The evidence. |
| Classification | firm_rule, example or guidance. |
| Review status | Verified, Needs review, or Not a firm rule. |
| Notes | Warnings: **CONFLICT** (another value exists for the same limit, with its source), "also stated in" (the same rule appears twice), "negative value: check the system can accept it", or the month-only warning. |

*Amber rows* need a person's attention. **"Confirm with policy owner" means the sentence does not say whether the limit itself is allowed. The tool never guesses.**

**Findings**: everything extracted, including values that failed the checks.
**Search log**: for each field, the phrasings searched, where they came from, the paragraphs read (with similarity), attempts, and step timings.
**Read me**: this explanation, inside the file.

To get example or guidance rows in the Test values sheet, download with `?include_examples=true` on the export address. To change the step size (default is one unit of the last decimal place, so 6.5 gives 6.4 and 6.6), add `&step=0.5`.

## 7. If something goes wrong
- **Ollama: not reachable**: start Ollama (`ollama serve`) and reload.
- **A question times out**: raise `OLLAMA_TIMEOUT` in `.env` (see `docs/DEPLOYMENT.md`).
- **Too slow**: tick Quick mode, close other programs, or ask your administrator about a smaller model.
- **"This conversation is no longer on the server"**: conversations are kept in memory for about 8 hours and are lost when the server restarts. Documents and batches are not.
