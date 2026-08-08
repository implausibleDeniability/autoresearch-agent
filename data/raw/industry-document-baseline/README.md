# Industry Documents baseline

This offline dataset contains 21 documents from the Industry Documents collection:

- 17 McKinsey Documents
- 4 NY Times EPA Collection documents

## Contents

- `ground_truth_pii.csv`: document-level PII ground truth for all 21 documents. Empty answers are intentional no-PII examples.
- `ground_truth_people.json`: deduplicated person-level ground truth for documents containing labelled people.
- `texts/<document_id>.txt`: one OCR text file per document.

The text files are the Industry Documents OCR text used when the ground truth was created. They are stored locally and do not require Beagle Azure access. Source PDFs are not included.
