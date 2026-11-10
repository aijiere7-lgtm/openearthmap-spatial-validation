# Publication status

**PENDING — public release authorised, remote publication to be verified.**

| Item | State |
|---|---|
| Local Git repository, initial commit and v1.0.0 tag | Local |
| GitHub authentication | Read-only account lookup succeeded |
| Public repository, push and GitHub Release | Authorised. Remote repository and Release not yet verified. |
| Zenodo GitHub integration | Not verified for a remote repository. Authorise the integration and enable the repository before creating the GitHub Release. |
| Zenodo software record, version DOI and concept DOI | Pending successful archiving |
| Independent derived-data record and DOI | Independent deposit pending. |
| Manuscript links and references | Pending verified repository links and real DOIs |

## Approved release scope

The author approved public release of the prepared repository, with Fengjun Jiang as sole software creator. The original software uses the previously proposed MIT licence. The release includes the derived metrics, class summaries, spatial diagnostics and exact tile centroid coordinates and memberships. Raw imagery, labels, weights and prediction rasters are excluded. Source dataset terms continue to apply. The Git commit uses a neutral archival identity. Repository access and archive identifiers are reported only after verification.

## Release and deposit sequence

1. Verify the approved licence, creator metadata and data scope. Ensure the release tag resolves to the approved revision.
2. Create the public GitHub repository, push the reviewed branch and tag, verify the remote tree, and connect the repository in [Zenodo's GitHub integration](https://zenodo.org/account/settings/github/) before publishing the GitHub Release. Zenodo can read supported metadata from CITATION.cff. A separate .zenodo.json is unnecessary for this staged release.
3. Publish GitHub Release v1.0.0 with the reviewed release notes. Verify the GitHub Release and that Zenodo preserved the expected files. Resolve the software version and concept DOIs to the correct records. If derived data are deposited separately, verify that record and its distinct DOI.
4. Add the verified repository and DOI links to README and CITATION.cff in a follow-up commit. Add the software version and separate data citations to the paper and recheck its anonymous copy, page layout and references. Keep software and data DOIs distinct.

[GitHub archiving instructions](https://docs.github.com/en/repositories/archiving-a-github-repository/referencing-and-citing-content) · [Zenodo software metadata](https://help.zenodo.org/docs/github/describe-software/) · [OpenEarthMap source licences](https://github.com/bao18/open_earth_map#license)
