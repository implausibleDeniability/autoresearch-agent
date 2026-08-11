# 87k-token development data

This directory contains 20 complete Industry Documents examples used for broader development
validation. Together they contain 87,454 `o200k_base` tokens and 49,953 words, including 92 labeled
people and 228 labeled PII values. Twelve documents are negative examples. There is no separate test
split yet.

`ground_truth.json` maps each document ID to its document-scoped list of canonical `PIIItem` values.
