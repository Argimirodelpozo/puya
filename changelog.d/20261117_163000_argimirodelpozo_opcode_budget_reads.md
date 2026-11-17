### Fixed

- Two reads of `global OpcodeBudget` in a row are no longer compiled as one read and a `dup`, and such a read is no longer moved past another load: each read gives a lower value than the one before.
