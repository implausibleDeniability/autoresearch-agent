# 202k-token development data

This dataset contains the 20 `dev-87k` Industry Documents plus 101 additional examples; `dev-87k`
remains available as a separate, cheaper dataset. Together these 121 documents contain 201,965
`o200k_base` tokens and 97,172 whitespace-delimited words, including 461 document-scoped people and
1,180 labeled PII values. Forty-five documents are negative examples. Sixty-three name components
recoverable only from an associated email local-part are marked optional for scoring.

`ground_truth.json` maps each document ID to its list of people. This expansion adds 101 documents
to the corrected original 20-document label set.

## Sampling

The additional documents come from public McKinsey collection records in the UCSF Industry
Documents Library. The source order used the Solr query `collectioncode:mck AND
availability_facet:public`, ordered by `random_20260813 asc`. The split spans emails, calendars,
presentations, documents, and spreadsheets, including long and layout-heavy sources.

## Label conventions

People are scoped to one document. Supported aliases, nicknames, and spelling variants for a person
are consolidated, while ambiguous partial names remain separate. Partial names, initials, and named
bibliography authors are retained. Contact details are attached only when the document associates
them with a person; shared meeting credentials, organizations, facilities, redactions, and template
placeholders are not labeled as people or person-owned PII.

## Sources beyond `dev-87k`

`Pages` is the rendered source page count, including workbook print pages rather than the one-page
ZIP metadata record.

| Document | Type | OCR tokens | Pages |
| --- | --- | ---: | ---: |
| [ssbf0256](https://www.industrydocuments.ucsf.edu/docs/#id=ssbf0256) | Calendar | 265 | 1 |
| [smjl0255](https://www.industrydocuments.ucsf.edu/docs/#id=smjl0255) | Calendar | 232 | 1 |
| [ghxg0257](https://www.industrydocuments.ucsf.edu/docs/#id=ghxg0257) | Presentation | 586 | 10 |
| [ykxp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ykxp0256) | Spreadsheet | 1,328 | 5 |
| [jgjw0256](https://www.industrydocuments.ucsf.edu/docs/#id=jgjw0256) | Email | 689 | 1 |
| [tlyd0257](https://www.industrydocuments.ucsf.edu/docs/#id=tlyd0257) | Calendar | 50 | 1 |
| [rxvd0256](https://www.industrydocuments.ucsf.edu/docs/#id=rxvd0256) | Email | 203 | 1 |
| [zmbw0256](https://www.industrydocuments.ucsf.edu/docs/#id=zmbw0256) | Email | 832 | 2 |
| [tqyk0256](https://www.industrydocuments.ucsf.edu/docs/#id=tqyk0256) | Spreadsheet | 18,847 | 40 |
| [fldm0255](https://www.industrydocuments.ucsf.edu/docs/#id=fldm0255) | Calendar | 190 | 1 |
| [yqpp0256](https://www.industrydocuments.ucsf.edu/docs/#id=yqpp0256) | Presentation | 4,027 | 47 |
| [hphc0257](https://www.industrydocuments.ucsf.edu/docs/#id=hphc0257) | Calendar | 50 | 1 |
| [flhj0256](https://www.industrydocuments.ucsf.edu/docs/#id=flhj0256) | Presentation | 4,762 | 56 |
| [zkfm0255](https://www.industrydocuments.ucsf.edu/docs/#id=zkfm0255) | Calendar | 171 | 1 |
| [nxmc0256](https://www.industrydocuments.ucsf.edu/docs/#id=nxmc0256) | Email | 345 | 4 |
| [llyw0256](https://www.industrydocuments.ucsf.edu/docs/#id=llyw0256) | Calendar | 293 | 1 |
| [skln0255](https://www.industrydocuments.ucsf.edu/docs/#id=skln0255) | Document | 3,634 | 4 |
| [qppf0257](https://www.industrydocuments.ucsf.edu/docs/#id=qppf0257) | Document | 782 | 35 |
| [hkjv0256](https://www.industrydocuments.ucsf.edu/docs/#id=hkjv0256) | Email | 210 | 1 |
| [lplw0257](https://www.industrydocuments.ucsf.edu/docs/#id=lplw0257) | Calendar | 138 | 1 |
| [kqbh0256](https://www.industrydocuments.ucsf.edu/docs/#id=kqbh0256) | Email | 1,005 | 2 |
| [nzdy0255](https://www.industrydocuments.ucsf.edu/docs/#id=nzdy0255) | Calendar | 247 | 1 |
| [nyxy0255](https://www.industrydocuments.ucsf.edu/docs/#id=nyxy0255) | Calendar | 277 | 1 |
| [rmyl0255](https://www.industrydocuments.ucsf.edu/docs/#id=rmyl0255) | Calendar | 214 | 1 |
| [lmhv0256](https://www.industrydocuments.ucsf.edu/docs/#id=lmhv0256) | Email | 159 | 1 |
| [nzcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=nzcx0256) | Calendar | 249 | 1 |
| [fqdd0257](https://www.industrydocuments.ucsf.edu/docs/#id=fqdd0257) | Calendar | 50 | 1 |
| [hnwg0256](https://www.industrydocuments.ucsf.edu/docs/#id=hnwg0256) | Email | 280 | 1 |
| [tkww0257](https://www.industrydocuments.ucsf.edu/docs/#id=tkww0257) | Spreadsheet | 57 | 1 |
| [mxvn0255](https://www.industrydocuments.ucsf.edu/docs/#id=mxvn0255) | Calendar | 222 | 1 |
| [rqkm0256](https://www.industrydocuments.ucsf.edu/docs/#id=rqkm0256) | Presentation | 3,522 | 20 |
| [rlcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=rlcx0256) | Email | 249 | 1 |
| [xlyn0256](https://www.industrydocuments.ucsf.edu/docs/#id=xlyn0256) | Presentation | 1,044 | 20 |
| [mfkx0256](https://www.industrydocuments.ucsf.edu/docs/#id=mfkx0256) | Presentation | 3,241 | 38 |
| [grdl0256](https://www.industrydocuments.ucsf.edu/docs/#id=grdl0256) | Calendar | 160 | 1 |
| [hmbd0257](https://www.industrydocuments.ucsf.edu/docs/#id=hmbd0257) | Calendar | 50 | 1 |
| [xnww0257](https://www.industrydocuments.ucsf.edu/docs/#id=xnww0257) | Presentation | 473 | 9 |
| [lzln0255](https://www.industrydocuments.ucsf.edu/docs/#id=lzln0255) | Email | 1,393 | 4 |
| [fkmg0256](https://www.industrydocuments.ucsf.edu/docs/#id=fkmg0256) | Email | 2,112 | 5 |
| [ltfv0257](https://www.industrydocuments.ucsf.edu/docs/#id=ltfv0257) | Calendar | 50 | 1 |
| [sxvb0257](https://www.industrydocuments.ucsf.edu/docs/#id=sxvb0257) | Calendar | 50 | 1 |
| [pkjx0256](https://www.industrydocuments.ucsf.edu/docs/#id=pkjx0256) | Spreadsheet | 4,558 | 9 |
| [qpxy0256](https://www.industrydocuments.ucsf.edu/docs/#id=qpxy0256) | Spreadsheet | 50 | 1 |
| [hpmc0256](https://www.industrydocuments.ucsf.edu/docs/#id=hpmc0256) | Calendar | 202 | 1 |
| [rmcx0256](https://www.industrydocuments.ucsf.edu/docs/#id=rmcx0256) | Email | 299 | 1 |
| [qtnv0257](https://www.industrydocuments.ucsf.edu/docs/#id=qtnv0257) | Calendar | 50 | 1 |
| [hnwf0256](https://www.industrydocuments.ucsf.edu/docs/#id=hnwf0256) | Calendar | 605 | 1 |
| [ghml0255](https://www.industrydocuments.ucsf.edu/docs/#id=ghml0255) | Calendar | 482 | 1 |
| [qtyy0255](https://www.industrydocuments.ucsf.edu/docs/#id=qtyy0255) | Email | 1,844 | 6 |
| [xsbg0257](https://www.industrydocuments.ucsf.edu/docs/#id=xsbg0257) | Spreadsheet | 184 | 1 |
| [nqjy0256](https://www.industrydocuments.ucsf.edu/docs/#id=nqjy0256) | Calendar | 44 | 1 |
| [smyl0255](https://www.industrydocuments.ucsf.edu/docs/#id=smyl0255) | Calendar | 208 | 1 |
| [tgll0255](https://www.industrydocuments.ucsf.edu/docs/#id=tgll0255) | Calendar | 240 | 1 |
| [rkml0255](https://www.industrydocuments.ucsf.edu/docs/#id=rkml0255) | Email | 206 | 1 |
| [jxkm0255](https://www.industrydocuments.ucsf.edu/docs/#id=jxkm0255) | Calendar | 746 | 2 |
| [qphc0257](https://www.industrydocuments.ucsf.edu/docs/#id=qphc0257) | Calendar | 149 | 1 |
| [hpjd0256](https://www.industrydocuments.ucsf.edu/docs/#id=hpjd0256) | Calendar | 123 | 1 |
| [thxf0256](https://www.industrydocuments.ucsf.edu/docs/#id=thxf0256) | Email | 386 | 1 |
| [xyjv0257](https://www.industrydocuments.ucsf.edu/docs/#id=xyjv0257) | Calendar | 50 | 1 |
| [rrmb0256](https://www.industrydocuments.ucsf.edu/docs/#id=rrmb0256) | Email | 1,011 | 2 |
| [tnmv0256](https://www.industrydocuments.ucsf.edu/docs/#id=tnmv0256) | Email | 804 | 2 |
| [ljpv0257](https://www.industrydocuments.ucsf.edu/docs/#id=ljpv0257) | Calendar | 50 | 1 |
| [ynbp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ynbp0256) | Presentation | 945 | 10 |
| [pxhd0256](https://www.industrydocuments.ucsf.edu/docs/#id=pxhd0256) | Email | 73 | 1 |
| [fmpm0255](https://www.industrydocuments.ucsf.edu/docs/#id=fmpm0255) | Email | 244 | 1 |
| [ngyn0256](https://www.industrydocuments.ucsf.edu/docs/#id=ngyn0256) | Presentation | 5,981 | 40 |
| [tmfy0256](https://www.industrydocuments.ucsf.edu/docs/#id=tmfy0256) | Spreadsheet | 7,846 | 30 |
| [fslf0256](https://www.industrydocuments.ucsf.edu/docs/#id=fslf0256) | Email | 632 | 2 |
| [mxhg0256](https://www.industrydocuments.ucsf.edu/docs/#id=mxhg0256) | Email | 207 | 1 |
| [khfw0256](https://www.industrydocuments.ucsf.edu/docs/#id=khfw0256) | Calendar | 530 | 2 |
| [ypvp0256](https://www.industrydocuments.ucsf.edu/docs/#id=ypvp0256) | Presentation | 1,163 | 10 |
| [qfxg0256](https://www.industrydocuments.ucsf.edu/docs/#id=qfxg0256) | Email | 681 | 2 |
| [jfkf0256](https://www.industrydocuments.ucsf.edu/docs/#id=jfkf0256) | Email | 300 | 1 |
| [kjck0256](https://www.industrydocuments.ucsf.edu/docs/#id=kjck0256) | Spreadsheet | 2,335 | 5 |
| [fxpn0255](https://www.industrydocuments.ucsf.edu/docs/#id=fxpn0255) | Email | 1,443 | 3 |
| [lqbw0257](https://www.industrydocuments.ucsf.edu/docs/#id=lqbw0257) | Calendar | 50 | 1 |
| [gnpv0257](https://www.industrydocuments.ucsf.edu/docs/#id=gnpv0257) | Calendar | 50 | 1 |
| [hxyj0256](https://www.industrydocuments.ucsf.edu/docs/#id=hxyj0256) | Presentation | 2,171 | 14 |
| [lxcm0256](https://www.industrydocuments.ucsf.edu/docs/#id=lxcm0256) | Spreadsheet | 73 | 1 |
| [rkhw0256](https://www.industrydocuments.ucsf.edu/docs/#id=rkhw0256) | Calendar | 233 | 1 |
| [pyvg0256](https://www.industrydocuments.ucsf.edu/docs/#id=pyvg0256) | Email | 884 | 2 |
| [yjgy0256](https://www.industrydocuments.ucsf.edu/docs/#id=yjgy0256) | Calendar | 112 | 1 |
| [mtjb0257](https://www.industrydocuments.ucsf.edu/docs/#id=mtjb0257) | Calendar | 50 | 1 |
| [lhbw0256](https://www.industrydocuments.ucsf.edu/docs/#id=lhbw0256) | Calendar | 745 | 2 |
| [mydb0256](https://www.industrydocuments.ucsf.edu/docs/#id=mydb0256) | Calendar | 121 | 1 |
| [fhhh0256](https://www.industrydocuments.ucsf.edu/docs/#id=fhhh0256) | Email | 700 | 1 |
| [hhwv0257](https://www.industrydocuments.ucsf.edu/docs/#id=hhwv0257) | Calendar | 50 | 1 |
| [rnhl0255](https://www.industrydocuments.ucsf.edu/docs/#id=rnhl0255) | Calendar | 241 | 1 |
| [glfv0257](https://www.industrydocuments.ucsf.edu/docs/#id=glfv0257) | Calendar | 50 | 1 |
| [ggxd0256](https://www.industrydocuments.ucsf.edu/docs/#id=ggxd0256) | Email | 174 | 1 |
| [zldc0256](https://www.industrydocuments.ucsf.edu/docs/#id=zldc0256) | Email | 189 | 1 |
| [nrhc0256](https://www.industrydocuments.ucsf.edu/docs/#id=nrhc0256) | Email | 216 | 1 |
| [xspk0256](https://www.industrydocuments.ucsf.edu/docs/#id=xspk0256) | Calendar | 51 | 1 |
| [lzmb0256](https://www.industrydocuments.ucsf.edu/docs/#id=lzmb0256) | Email | 303 | 1 |
| [jzhg0257](https://www.industrydocuments.ucsf.edu/docs/#id=jzhg0257) | Presentation | 1,214 | 16 |
| [qzmx0256](https://www.industrydocuments.ucsf.edu/docs/#id=qzmx0256) | Presentation | 1,598 | 20 |
| [flpw0256](https://www.industrydocuments.ucsf.edu/docs/#id=flpw0256) | Calendar | 341 | 1 |
| [njjp0256](https://www.industrydocuments.ucsf.edu/docs/#id=njjp0256) | Presentation | 577 | 2 |
| [rzkw0256](https://www.industrydocuments.ucsf.edu/docs/#id=rzkw0256) | Calendar | 1,693 | 3 |
| [lygw0256](https://www.industrydocuments.ucsf.edu/docs/#id=lygw0256) | Calendar | 141 | 1 |
| [qpgp0256](https://www.industrydocuments.ucsf.edu/docs/#id=qpgp0256) | Presentation | 15,896 | 1 |
