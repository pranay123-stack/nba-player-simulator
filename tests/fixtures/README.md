# Test fixtures (real data)

`real_boxscores_2022-23_subset.csv.gz` is an unmodified slice of the raw source rows for every
regular-season game from 2022-10-18 to 2023-01-15 (all 30 teams), in the source's own column
format. Source: github.com/NocturneBear/NBA-Data-2010-2024 at commit
`a5f108b5b1f08074d78b9e8e901926a9ce4c06c5` (MIT licence, see `data/raw/LICENSE` after download);
underlying data NBA.com `boxscoretraditionalv3`.

No synthetic datasets are used in the test suite. Individual unit tests contain small hand-written
values (e.g. a 3-row PMF) purely as assertions on arithmetic.
