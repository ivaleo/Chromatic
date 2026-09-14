# arxiv-en/ — short English version for arXiv

`bounds-en.tex` — *New upper bounds for the chromatic numbers of Euclidean
spaces: χ(ℝ⁴) ≤ 43, χ(ℝ⁵) ≤ 132, χ(ℝ⁷) ≤ 1029, χ(ℝ⁹) ≤ 7203, χ(ℝ¹⁰) ≤ 45619*.
Self-contained, English only, **8 pages** (10 pt, A4, no figures). Build:
`latexmk -pdf bounds-en.tex` → `bounds-en.pdf`. Decision of 14.09.2026: this
is the document intended for arXiv while the two full Russian articles
(`../article1`, `../article2`) are still being proofread. A first draft was
15 pages; on the author's request it was condensed to half without dropping
any statement, datum or proof: what went were the historical remarks, the
duplicated explanations of the protocol in each proof, the second
independent checks (kept as one sentence each), the bcc proof (kept as a
remark) and six references.

## Contents

1. Introduction: problem, previous bounds (ABPR), the refuted expectation,
   Theorem 1.1 with the table of five bounds, the two proof schemes.
2. Width and the certificate protocol: criterion, midpoint lemma, window
   2(1+ℓ)R, Proposition 2.3 (six items) with a short proof.
3. Four explicit constructions with all data: Eisenstein family and the
   index-43 lattice (Q, T, invariants, margin, independent lower bound);
   index 132 (Q, minors, φ, R², D², margin); laminations 1029
   (Q = (¾M, g; gᵀ, 17897/10⁴), C) and 7203 (Q = (³⁄₂A, g; gᵀ, 17982/10⁴), C)
   with facet/vertex counts, windows, minimizers and margins.
4. The Eisenstein identity (both halves, with proofs), exact widths of the
   7^{n/2} constructions, K₁₂; the product rule; 45619, 19·7¹², 9604, 4·7¹².
5. The index floor on E₈ (2401 unimprovable, full proof); bcc, E₆*, Leech
   in one remark.
6. Reproducibility, status of claims, contributions and the use of AI.

Everything stated is a theorem or an exact certificate; the numerical
candidate 28812 is mentioned once, explicitly as not claimed.

## Correspondence with the certificates

All fractions and matrices are copied from `../../audit-data/results/`:
`dim4_k43_verify.json`, `metric_deform_a5_132_refined_certificate.json`,
`dim7_1029_exact.json`, `dim9_7203_exact.json`. The test
`audit-data/tests/test_paper_short_versions.py` checks that the block
matrices (A, M, g, C), the covering radii, the minimal distances, the
normalized widths and the margins printed here coincide with the JSON files.
Note: in `dim7_1029_exact.json` the minimizers are 27 **pairs** ±v (the images
of the 54 minimal vectors of E₆*), which is how this text states it.

## arXiv metadata (draft; the submitting author confirms)

- **Title:** New upper bounds for the chromatic numbers of Euclidean spaces:
  chi(R^4) <= 43, chi(R^5) <= 132, chi(R^7) <= 1029, chi(R^9) <= 7203,
  chi(R^10) <= 45619
- **Authors:** Leonid L. Ivanov, Nadezhda V. Glushkova (order by
  contribution; affiliations — TODO, see `\author` in the source).
- **Abstract:** the abstract of the PDF (≈1400 characters, under the arXiv
  limit of 1920).
- **Categories:** math.MG (primary), math.CO (cross-list).
- **MSC:** 52C10 (primary); 05C15, 52C07, 11H31.
- **Comments:** 8 pages. Code, exact certificates and data:
  https://github.com/ivaleo/Chromatic
- **License:** arXiv non-exclusive license v1.0.
- Account, endorsement and the checklist: `../arxiv-metadata.md` (items
  1–3, 5–7 apply verbatim; the tarball is the single file `bounds-en.tex`,
  no figures).

## Relation to the other documents

The Russian articles remain the detailed versions (search algorithms,
ladders, screens, tilings); the UMN note carries the four certificates only,
the Doklady note (`../note-dan`) the analytic part only. This text
overlaps with both by design — arXiv is a preprint server, not a journal —
but the authors should check the preprint policy of the target journals
before posting (Doklady: `../note-dan/README.md`, last item).
