# Zenodo Upload Order

Version: `1.0.0`

Production run: `q1_cfm4_20260525_042149`

## Publish Two Linked Records

Software and derived results are both significant research outputs, so publish
them as two Zenodo records.

### Record A: Software

For the prepared files in this folder, use the manual route: create a Zenodo
**Software** upload and upload exactly:

1. `01_smear_autofocus_preprocessing_software_v1.0.0.tar.gz`

Keeping one compressed source archive allows Zenodo's software-preservation
workflow to process the record. Do not upload the result archives to this
software record.

Alternative: connect the GitHub repository to Zenodo and let Zenodo archive
GitHub release `v1.0.0`. If you choose automatic GitHub integration, do not also
publish the manual software archive as a second record for the same version.

Use metadata from `ZENODO_SOFTWARE_METADATA.md`.

### Record B: Complete Derived Results

Create a separate Zenodo **Dataset** upload. Upload in this order:

1. `00_RESULTS_RECORD_README.md`
2. `01_q1_cfm4_results_core_v1.0.0.tar.gz`
3. `02_q1_cfm4_results_large_csv_v1.0.0.tar.gz`
4. `03_q1_cfm4_execution_logs_v1.0.0.tar.gz`
5. `ARCHIVE_MANIFEST.csv`
6. `SHA256SUMS`

Select `00_RESULTS_RECORD_README.md` as the default preview. Use metadata from
`ZENODO_RESULTS_METADATA.md`.

## Required Actions Before Upload

1. Replace all `REPLACE_WITH_...` placeholders in the GitHub release metadata.
2. Add every real creator in publication order with ORCID and affiliation.
3. Add funding only when an actual grant funded the work.
4. Confirm that derived outputs may be redistributed under CC BY 4.0.
5. Confirm that no source microscopy images, patient identifiers, credentials,
   or restricted labels are present.
6. Verify archives with `sha256sum -c SHA256SUMS`.
7. Test the workflow first on `https://sandbox.zenodo.org` if desired.

After replacing GitHub metadata placeholders, rebuild the software archive and
its checksum with:

```bash
./PreprocessingZenodo/finalize_software_release.sh
```

## DOI Sequence

1. Create both Zenodo drafts.
2. Reserve both DOIs before publication.
3. Add the results DOI to the software record as a related work.
4. Add the software DOI to the results record as `Is derived from` or the
   nearest available equivalent.
5. Add both reserved DOIs to `CITATION.cff`, `.zenodo.json`, and the GitHub
   README before making the final GitHub release.
6. Publish the manual software record.
7. Publish the results record.
8. Create GitHub release `v1.0.0` and include both Zenodo DOI links. Do not
   enable automatic Zenodo ingestion for this release when using the manual
   software record.
9. Add journal article DOIs later as `Is supplement to` relations; do not use a
   journal article DOI as the DOI of either Zenodo upload.

After publication, metadata can be edited, but file changes should normally be
released as a new Zenodo version. Keep the version-specific DOI for exact
reproduction and the concept DOI for citing the evolving project.
