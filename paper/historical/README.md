# Historical SSRN source (not current results)

Recovered byte-for-byte with `git show` from commit
`d99064367b144a147457bc67ab992e599e3b5c25`, paths `analysis/paper/`.
The complete requested title, six figure sections, bibliography and mathematical
appendices occur in `main.tex`. The separately stored appendix/assumption files
are also preserved, although this version of main.tex contains its own appendices.

This snapshot contains obsolete empirical claims and invalid mathematical steps.
It is retained for the revision audit, not as a build target or source of current
results. The original historical graphs remain in git history; the six current
replacements are in `paper/figures/v2/`. No historical source was overwritten.

Discovery: fetched all remotes with pruning, inspected all branches and the full
history of `*.tex`, and searched all reachable revisions for both parts of the
requested title. Current `analysis/paper/main.tex` is a shorter implementation
note. The empty files formerly in `paper/` and `arxiv_submission/` were not used
as sources.
