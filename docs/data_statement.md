# Data Statement

## What is shared

`data/observed_global_counts.csv` contains aggregate annual counts of hand-surgery PubMed records that:

- were classified as in-scope by the manuscript's rules-based taxonomy, and
- received a non-empty country attribution.

Columns:

- `year` — calendar year of publication.
- `hand_surgery_publications` — count of records satisfying the inclusion criteria.
- `pubmed_retrieved_date` — date of the PubMed export used to produce the counts.

This file is the sole input required to reproduce all forecast outputs in this repository.

## What is not shared

The following are intentionally excluded from this repository:

- Raw PubMed exports (article-level records, abstracts, author metadata).
- Country-attributed article-level tables.
- The rules-based classifier and its taxonomy configuration.
- Reviewer-classified gold-set workbooks.
- Local language-model artifacts used for downstream country/topic enrichment.

Reasons: these files include manuscript-author working notes and country-resolution intermediate states that are not appropriate for public release. Aggregate counts are sufficient to reproduce the forecasting analysis; they do not enable reconstruction of the article-level dataset.

## Reproducing aggregation

The aggregation logic — including the in-scope filter and country requirement — is described in the manuscript's methods section. Researchers wishing to reproduce the aggregation from a fresh PubMed query are encouraged to contact the authors for the classification rule set.
