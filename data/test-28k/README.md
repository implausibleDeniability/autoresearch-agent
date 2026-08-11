# 28k-token manually labeled Industry Documents data

This directory contains thirty-six complete, previously unused documents from the UCSF Industry
Documents Library's McKinsey Documents collection. Together they contain 27,650 `o200k_base`
tokens and 9,992 words across 68 pages, including 336 labeled people and 906 labeled PII values.
Two slide decks are intentional no-person examples. Short iMessage exports and emails with large
recipient lists are included intentionally rather than filtered out for labeling difficulty.

`ground_truth.json` maps each document ID to its document-scoped list of canonical `PIIItem`
values. `texts/` contains the corresponding UCSF OCR text with line endings normalized to LF and
trailing whitespace removed.
The source PDFs are not included.

## Sources

| Document | Type | Pages | Labeled people | UCSF record |
| --- | --- | ---: | ---: | --- |
| `fpfb0256` | Email | 2 | 13 | [record](https://www.industrydocuments.ucsf.edu/docs/fpfb0256) |
| `tzvb0256` | Email | 2 | 7 | [record](https://www.industrydocuments.ucsf.edu/docs/tzvb0256) |
| `pybm0255` | Email | 2 | 11 | [record](https://www.industrydocuments.ucsf.edu/docs/pybm0255) |
| `mlkb0256` | Email | 2 | 10 | [record](https://www.industrydocuments.ucsf.edu/docs/mlkb0256) |
| `kgwl0255` | Email | 2 | 8 | [record](https://www.industrydocuments.ucsf.edu/docs/kgwl0255) |
| `khhb0256` | Email | 2 | 8 | [record](https://www.industrydocuments.ucsf.edu/docs/khhb0256) |
| `jrnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/jrnp0256) |
| `ssnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/ssnp0256) |
| `fldk0256` | Presentation | 6 | 0 | [record](https://www.industrydocuments.ucsf.edu/docs/fldk0256) |
| `tjdk0256` | Presentation | 2 | 0 | [record](https://www.industrydocuments.ucsf.edu/docs/tjdk0256) |
| `fmhb0256` | Email | 2 | 7 | [record](https://www.industrydocuments.ucsf.edu/docs/fmhb0256) |
| `gqfb0256` | Email | 2 | 8 | [record](https://www.industrydocuments.ucsf.edu/docs/gqfb0256) |
| `grfb0256` | Email | 2 | 15 | [record](https://www.industrydocuments.ucsf.edu/docs/grfb0256) |
| `rxgb0256` | Email | 2 | 12 | [record](https://www.industrydocuments.ucsf.edu/docs/rxgb0256) |
| `szxb0256` | Email | 2 | 11 | [record](https://www.industrydocuments.ucsf.edu/docs/szxb0256) |
| `hhkb0256` | Email and appointment | 2 | 12 | [record](https://www.industrydocuments.ucsf.edu/docs/hhkb0256) |
| `lfml0255` | Email | 2 | 10 | [record](https://www.industrydocuments.ucsf.edu/docs/lfml0255) |
| `jxmm0255` | Email | 2 | 7 | [record](https://www.industrydocuments.ucsf.edu/docs/jxmm0255) |
| `pzkv0256` | Email | 2 | 6 | [record](https://www.industrydocuments.ucsf.edu/docs/pzkv0256) |
| `jgkv0256` | Email | 2 | 10 | [record](https://www.industrydocuments.ucsf.edu/docs/jgkv0256) |
| `gkkb0256` | Email and appointment | 3 | 9 | [record](https://www.industrydocuments.ucsf.edu/docs/gkkb0256) |
| `nzkv0256` | Email | 2 | 24 | [record](https://www.industrydocuments.ucsf.edu/docs/nzkv0256) |
| `szgk0255` | Email | 2 | 12 | [record](https://www.industrydocuments.ucsf.edu/docs/szgk0255) |
| `xkmv0256` | Email | 2 | 25 | [record](https://www.industrydocuments.ucsf.edu/docs/xkmv0256) |
| `xlkm0255` | Email | 2 | 9 | [record](https://www.industrydocuments.ucsf.edu/docs/xlkm0255) |
| `zqlv0256` | Email | 2 | 11 | [record](https://www.industrydocuments.ucsf.edu/docs/zqlv0256) |
| `fmbl0255` | Email | 3 | 36 | [record](https://www.industrydocuments.ucsf.edu/docs/fmbl0255) |
| `jnwx0256` | Email | 2 | 34 | [record](https://www.industrydocuments.ucsf.edu/docs/jnwx0256) |
| `fsnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/fsnp0256) |
| `lsnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/lsnp0256) |
| `mrnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/mrnp0256) |
| `nrnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/nrnp0256) |
| `qrnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/qrnp0256) |
| `rsnp0256` | iMessage export | 1 | 3 | [record](https://www.industrydocuments.ucsf.edu/docs/rsnp0256) |
| `trnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/trnp0256) |
| `yrnp0256` | iMessage export | 1 | 2 | [record](https://www.industrydocuments.ucsf.edu/docs/yrnp0256) |

## Labeling protocol

Every PDF page was reviewed visually against both the UCSF OCR and a separate local OCR pass.
Labels include people in message metadata, recipient lists, body text, signatures, appointment
details, and participant lists. Repeated references to the same person within a document are
consolidated into one canonical person. Explicit email addresses, phone numbers, and locations
are attached to that person; conference dial-ins and fully redacted values are excluded.

Names recoverable from explicit email addresses are labeled even when the display name is absent.
Honorifics, degrees, job titles, organization names, and redacted names are not labels. A partial
name is retained only when the document provides no reliable expansion. No hidden redacted value
was inferred.
