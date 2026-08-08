# Development split

This split contains four deliberately selected documents from the Industry Documents baseline:

- `fjwg0257`: a redacted McKinsey document with no labelled PII.
- `pzvv0257`: a McKinsey document with person-like text but no labelled PII.
- `nrbn0226`: a noisy NY Times EPA document labelled with names, locations, phone numbers, and work emails.
- `rtbn0226`: a related EPA document that adds middle-name coverage and keeps shared people within one split.

The selection balances positive and negative examples while covering both source collections and every field type present in the labels. Documents linked by a labelled person are kept in the same split to prevent person-level leakage.

`ground_truth.json` maps each document ID to its document-scoped list of canonical `PIIItem` values.
