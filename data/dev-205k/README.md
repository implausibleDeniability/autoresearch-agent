# 205k-token development data

This dataset contains the 20 `dev-87k` Industry Documents plus 102 new examples; `dev-87k` remains
available as a separate, cheaper dataset. Together these 122 documents contain 204,153 `o200k_base`
tokens and 98,127 whitespace-delimited words,
including 534 document-scoped people and 1,334 labeled PII values. Forty-six documents are
negative examples.

`ground_truth.json` maps each document ID to its list of canonical `PIIItem` values. The original
20 documents and their labels are unchanged; this expansion adds 102 newly annotated documents.

## Sampling

The added documents are a reproducible random sample of public McKinsey collection records from
the UCSF Industry Documents Library. The Solr query used `collectioncode:mck AND
availability_facet:public`, ordered by `random_20260813 asc`. IDs already present anywhere in this
repository's datasets—including blind test data—were removed before selection. The queue was then
accepted in order until the combined dataset exceeded 200,000 source tokens.

The only routine exclusion was a document exceeding 20,000 `o200k_base` OCR tokens. Dense or
sparse content, no PII, OCR errors, repetition, grids, spreadsheets, and difficult layouts were not
exclusion reasons. The sample therefore includes an 18,847-token pricing workbook, bibliography
grids, redacted pages, low-information calendars, long presentations, and OCR-damaged email.

### Excluded records

| Queue | Document | OCR tokens | Specific reason |
| ---: | --- | ---: | --- |
| 9 | [lsmx0256](https://www.industrydocuments.ucsf.edu/docs/#id=lsmx0256) | 9,207,446 | Exceeded the 20,000-token limit. |
| 21 | [lgyj0256](https://www.industrydocuments.ucsf.edu/docs/#id=lgyj0256) | 280,967 | Exceeded the 20,000-token limit. |
| 75 | [pypj0256](https://www.industrydocuments.ucsf.edu/docs/#id=pypj0256) | 27,589 | Exceeded the 20,000-token limit. |

No other sampled record was excluded.

## Annotation and validation

Each source OCR file was checked against its PDF or extracted workbook. All 555 rendered PDF pages
and workbook print pages were reviewed; an independent local OCR pass was used to surface names
that the source OCR or layout could obscure. Presentation, grid, and spreadsheet pages were also
inspected visually. Every added label value was then verified as a literal, case-insensitive source
substring.

People are scoped to one document. Supported aliases, nicknames, and spelling variants for a person
are consolidated, while ambiguous partial names remain separate. Partial names, initials, and named
bibliography authors are retained. Contact details are attached only when the document associates
them with a person; shared meeting credentials, organizations, facilities, redactions, and template
placeholders are not labeled as people or person-owned PII.

## Added sources

`Pages` is the number of rendered pages reviewed, including workbook print pages rather than the
one-page ZIP metadata record.

| Queue | Document | Type | OCR tokens | Pages |
| ---: | --- | --- | ---: | ---: |
| 1 | [ssbf0256](https://www.industrydocuments.ucsf.edu/docs/#id=ssbf0256) | Calendar | 265 | 1 |
| 2 | [smjl0255](https://www.industrydocuments.ucsf.edu/docs/#id=smjl0255) | Calendar | 232 | 1 |
| 3 | [ghxg0257](https://www.industrydocuments.ucsf.edu/docs/#id=ghxg0257) | Presentation | 586 | 10 |
| 4 | [ykxp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ykxp0256) | Spreadsheet | 1,328 | 5 |
| 5 | [jgjw0256](https://www.industrydocuments.ucsf.edu/docs/#id=jgjw0256) | Email | 689 | 1 |
| 6 | [tlyd0257](https://www.industrydocuments.ucsf.edu/docs/#id=tlyd0257) | Calendar | 50 | 1 |
| 7 | [rxvd0256](https://www.industrydocuments.ucsf.edu/docs/#id=rxvd0256) | Email | 203 | 1 |
| 8 | [zmbw0256](https://www.industrydocuments.ucsf.edu/docs/#id=zmbw0256) | Email | 832 | 2 |
| 10 | [tqyk0256](https://www.industrydocuments.ucsf.edu/docs/#id=tqyk0256) | Spreadsheet | 18,847 | 40 |
| 11 | [fldm0255](https://www.industrydocuments.ucsf.edu/docs/#id=fldm0255) | Calendar | 190 | 1 |
| 12 | [yqpp0256](https://www.industrydocuments.ucsf.edu/docs/#id=yqpp0256) | Presentation | 4,027 | 47 |
| 13 | [hphc0257](https://www.industrydocuments.ucsf.edu/docs/#id=hphc0257) | Calendar | 50 | 1 |
| 14 | [flhj0256](https://www.industrydocuments.ucsf.edu/docs/#id=flhj0256) | Presentation | 4,762 | 56 |
| 15 | [zkfm0255](https://www.industrydocuments.ucsf.edu/docs/#id=zkfm0255) | Calendar | 171 | 1 |
| 16 | [nxmc0256](https://www.industrydocuments.ucsf.edu/docs/#id=nxmc0256) | Email | 345 | 4 |
| 17 | [llyw0256](https://www.industrydocuments.ucsf.edu/docs/#id=llyw0256) | Calendar | 293 | 1 |
| 18 | [skln0255](https://www.industrydocuments.ucsf.edu/docs/#id=skln0255) | Document | 3,634 | 4 |
| 19 | [qppf0257](https://www.industrydocuments.ucsf.edu/docs/#id=qppf0257) | Document | 782 | 35 |
| 20 | [hkjv0256](https://www.industrydocuments.ucsf.edu/docs/#id=hkjv0256) | Email | 210 | 1 |
| 22 | [lplw0257](https://www.industrydocuments.ucsf.edu/docs/#id=lplw0257) | Calendar | 138 | 1 |
| 23 | [kqbh0256](https://www.industrydocuments.ucsf.edu/docs/#id=kqbh0256) | Email | 1,005 | 2 |
| 24 | [nzdy0255](https://www.industrydocuments.ucsf.edu/docs/#id=nzdy0255) | Calendar | 247 | 1 |
| 25 | [nyxy0255](https://www.industrydocuments.ucsf.edu/docs/#id=nyxy0255) | Calendar | 277 | 1 |
| 26 | [rmyl0255](https://www.industrydocuments.ucsf.edu/docs/#id=rmyl0255) | Calendar | 214 | 1 |
| 27 | [lmhv0256](https://www.industrydocuments.ucsf.edu/docs/#id=lmhv0256) | Email | 159 | 1 |
| 28 | [nzcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=nzcx0256) | Calendar | 249 | 1 |
| 29 | [fqdd0257](https://www.industrydocuments.ucsf.edu/docs/#id=fqdd0257) | Calendar | 50 | 1 |
| 30 | [hnwg0256](https://www.industrydocuments.ucsf.edu/docs/#id=hnwg0256) | Email | 280 | 1 |
| 31 | [tkww0257](https://www.industrydocuments.ucsf.edu/docs/#id=tkww0257) | Spreadsheet | 57 | 1 |
| 32 | [mxvn0255](https://www.industrydocuments.ucsf.edu/docs/#id=mxvn0255) | Calendar | 222 | 1 |
| 33 | [rqkm0256](https://www.industrydocuments.ucsf.edu/docs/#id=rqkm0256) | Presentation | 3,522 | 20 |
| 34 | [rlcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=rlcx0256) | Email | 249 | 1 |
| 35 | [xlyn0256](https://www.industrydocuments.ucsf.edu/docs/#id=xlyn0256) | Presentation | 1,044 | 20 |
| 36 | [mfkx0256](https://www.industrydocuments.ucsf.edu/docs/#id=mfkx0256) | Presentation | 3,241 | 38 |
| 37 | [grdl0256](https://www.industrydocuments.ucsf.edu/docs/#id=grdl0256) | Calendar | 160 | 1 |
| 38 | [hmbd0257](https://www.industrydocuments.ucsf.edu/docs/#id=hmbd0257) | Calendar | 50 | 1 |
| 39 | [xnww0257](https://www.industrydocuments.ucsf.edu/docs/#id=xnww0257) | Presentation | 473 | 9 |
| 40 | [lzln0255](https://www.industrydocuments.ucsf.edu/docs/#id=lzln0255) | Email | 1,393 | 4 |
| 41 | [fkmg0256](https://www.industrydocuments.ucsf.edu/docs/#id=fkmg0256) | Email | 2,112 | 5 |
| 42 | [ltfv0257](https://www.industrydocuments.ucsf.edu/docs/#id=ltfv0257) | Calendar | 50 | 1 |
| 43 | [sxvb0257](https://www.industrydocuments.ucsf.edu/docs/#id=sxvb0257) | Calendar | 50 | 1 |
| 44 | [pkjx0256](https://www.industrydocuments.ucsf.edu/docs/#id=pkjx0256) | Spreadsheet | 4,558 | 9 |
| 45 | [qpxy0256](https://www.industrydocuments.ucsf.edu/docs/#id=qpxy0256) | Spreadsheet | 50 | 1 |
| 46 | [hpmc0256](https://www.industrydocuments.ucsf.edu/docs/#id=hpmc0256) | Calendar | 202 | 1 |
| 47 | [rmcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=rmcx0256) | Email | 299 | 1 |
| 48 | [qtnv0257](https://www.industrydocuments.ucsf.edu/docs/#id=qtnv0257) | Calendar | 50 | 1 |
| 49 | [hnwf0256](https://www.industrydocuments.ucsf.edu/docs/#id=hnwf0256) | Calendar | 605 | 1 |
| 50 | [ghml0255](https://www.industrydocuments.ucsf.edu/docs/#id=ghml0255) | Calendar | 482 | 1 |
| 51 | [qtyy0255](https://www.industrydocuments.ucsf.edu/docs/#id=qtyy0255) | Email | 1,844 | 6 |
| 52 | [xsbg0257](https://www.industrydocuments.ucsf.edu/docs/#id=xsbg0257) | Spreadsheet | 184 | 1 |
| 53 | [nqjy0256](https://www.industrydocuments.ucsf.edu/docs/#id=nqjy0256) | Calendar | 44 | 1 |
| 54 | [smyl0255](https://www.industrydocuments.ucsf.edu/docs/#id=smyl0255) | Calendar | 208 | 1 |
| 55 | [tgll0255](https://www.industrydocuments.ucsf.edu/docs/#id=tgll0255) | Calendar | 240 | 1 |
| 56 | [rkml0255](https://www.industrydocuments.ucsf.edu/docs/#id=rkml0255) | Email | 206 | 1 |
| 57 | [jxkm0255](https://www.industrydocuments.ucsf.edu/docs/#id=jxkm0255) | Calendar | 746 | 2 |
| 58 | [qphc0257](https://www.industrydocuments.ucsf.edu/docs/#id=qphc0257) | Calendar | 149 | 1 |
| 59 | [hpjd0256](https://www.industrydocuments.ucsf.edu/docs/#id=hpjd0256) | Calendar | 123 | 1 |
| 60 | [thxf0256](https://www.industrydocuments.ucsf.edu/docs/#id=thxf0256) | Email | 386 | 1 |
| 61 | [xyjv0257](https://www.industrydocuments.ucsf.edu/docs/#id=xyjv0257) | Calendar | 50 | 1 |
| 62 | [rrmb0256](https://www.industrydocuments.ucsf.edu/docs/#id=rrmb0256) | Email | 1,011 | 2 |
| 63 | [tnmv0256](https://www.industrydocuments.ucsf.edu/docs/#id=tnmv0256) | Email | 804 | 2 |
| 64 | [gxkx0256](https://www.industrydocuments.ucsf.edu/docs/#id=gxkx0256) | Spreadsheet | 2,189 | 6 |
| 65 | [ljpv0257](https://www.industrydocuments.ucsf.edu/docs/#id=ljpv0257) | Calendar | 50 | 1 |
| 66 | [ynbp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ynbp0256) | Presentation | 945 | 10 |
| 67 | [pxhd0256](https://www.industrydocuments.ucsf.edu/docs/#id=pxhd0256) | Email | 73 | 1 |
| 68 | [fmpm0255](https://www.industrydocuments.ucsf.edu/docs/#id=fmpm0255) | Email | 244 | 1 |
| 69 | [ngyn0256](https://www.industrydocuments.ucsf.edu/docs/#id=ngyn0256) | Presentation | 5,981 | 40 |
| 70 | [tmfy0256](https://www.industrydocuments.ucsf.edu/docs/#id=tmfy0256) | Spreadsheet | 7,846 | 30 |
| 71 | [fslf0256](https://www.industrydocuments.ucsf.edu/docs/#id=fslf0256) | Email | 632 | 2 |
| 72 | [mxhg0256](https://www.industrydocuments.ucsf.edu/docs/#id=mxhg0256) | Email | 207 | 1 |
| 73 | [khfw0256](https://www.industrydocuments.ucsf.edu/docs/#id=khfw0256) | Calendar | 530 | 2 |
| 74 | [ypvp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ypvp0256) | Presentation | 1,163 | 10 |
| 76 | [qfxg0256](https://www.industrydocuments.ucsf.edu/docs/#id=qfxg0256) | Email | 681 | 2 |
| 77 | [jfkf0256](https://www.industrydocuments.ucsf.edu/docs/#id=jfkf0256) | Email | 300 | 1 |
| 78 | [kjck0256](https://www.industrydocuments.ucsf.edu/docs/#id=kjck0256) | Spreadsheet | 2,335 | 5 |
| 79 | [fxpn0255](https://www.industrydocuments.ucsf.edu/docs/#id=fxpn0255) | Email | 1,443 | 3 |
| 80 | [lqbw0257](https://www.industrydocuments.ucsf.edu/docs/#id=lqbw0257) | Calendar | 50 | 1 |
| 81 | [gnpv0257](https://www.industrydocuments.ucsf.edu/docs/#id=gnpv0257) | Calendar | 50 | 1 |
| 82 | [hxyj0256](https://www.industrydocuments.ucsf.edu/docs/#id=hxyj0256) | Presentation | 2,171 | 14 |
| 83 | [lxcm0256](https://www.industrydocuments.ucsf.edu/docs/#id=lxcm0256) | Spreadsheet | 73 | 1 |
| 84 | [rkhw0256](https://www.industrydocuments.ucsf.edu/docs/#id=rkhw0256) | Calendar | 233 | 1 |
| 85 | [pyvg0256](https://www.industrydocuments.ucsf.edu/docs/#id=pyvg0256) | Email | 884 | 2 |
| 86 | [yjgy0256](https://www.industrydocuments.ucsf.edu/docs/#id=yjgy0256) | Calendar | 112 | 1 |
| 87 | [mtjb0257](https://www.industrydocuments.ucsf.edu/docs/#id=mtjb0257) | Calendar | 50 | 1 |
| 88 | [lhbw0256](https://www.industrydocuments.ucsf.edu/docs/#id=lhbw0256) | Calendar | 745 | 2 |
| 89 | [mydb0256](https://www.industrydocuments.ucsf.edu/docs/#id=mydb0256) | Calendar | 121 | 1 |
| 90 | [fhhh0256](https://www.industrydocuments.ucsf.edu/docs/#id=fhhh0256) | Email | 700 | 1 |
| 91 | [hhwv0257](https://www.industrydocuments.ucsf.edu/docs/#id=hhwv0257) | Calendar | 50 | 1 |
| 92 | [rnhl0255](https://www.industrydocuments.ucsf.edu/docs/#id=rnhl0255) | Calendar | 241 | 1 |
| 93 | [glfv0257](https://www.industrydocuments.ucsf.edu/docs/#id=glfv0257) | Calendar | 50 | 1 |
| 94 | [ggxd0256](https://www.industrydocuments.ucsf.edu/docs/#id=ggxd0256) | Email | 174 | 1 |
| 95 | [zldc0256](https://www.industrydocuments.ucsf.edu/docs/#id=zldc0256) | Email | 189 | 1 |
| 96 | [nrhc0256](https://www.industrydocuments.ucsf.edu/docs/#id=nrhc0256) | Email | 216 | 1 |
| 97 | [xspk0256](https://www.industrydocuments.ucsf.edu/docs/#id=xspk0256) | Calendar | 51 | 1 |
| 98 | [lzmb0256](https://www.industrydocuments.ucsf.edu/docs/#id=lzmb0256) | Email | 303 | 1 |
| 99 | [jzhg0257](https://www.industrydocuments.ucsf.edu/docs/#id=jzhg0257) | Presentation | 1,214 | 16 |
| 100 | [qzmx0256](https://www.industrydocuments.ucsf.edu/docs/#id=qzmx0256) | Presentation | 1,598 | 20 |
| 101 | [flpw0256](https://www.industrydocuments.ucsf.edu/docs/#id=flpw0256) | Calendar | 341 | 1 |
| 102 | [njjp0256](https://www.industrydocuments.ucsf.edu/docs/#id=njjp0256) | Presentation | 577 | 2 |
| 103 | [rzkw0256](https://www.industrydocuments.ucsf.edu/docs/#id=rzkw0256) | Calendar | 1,693 | 3 |
| 104 | [lygw0256](https://www.industrydocuments.ucsf.edu/docs/#id=lygw0256) | Calendar | 141 | 1 |
| 105 | [qpgp0256](https://www.industrydocuments.ucsf.edu/docs/#id=qpgp0256) | Presentation | 15,896 | 1 |
