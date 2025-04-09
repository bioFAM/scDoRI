# scDoRI

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/bioFAM/scDoRI/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scDoRI

scDoRI (single-cell Deep multi-Omic Regulatory Inference) is a deep learning model to infer enhancer-mediated gene regulatory networks (eGRNs) from single-cell multiome data (RNA + ATAC).

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install scDoRI:

<!--
1) Install the latest release of `scDoRI` from [PyPI][]:

```bash
pip install scDoRI
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/bioFAM/scDoRI.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/bioFAM/scDoRI/issues
[tests]: https://github.com/bioFAM/scDoRI/actions/workflows/test.yaml
[documentation]: https://scDoRI.readthedocs.io
[changelog]: https://scDoRI.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scDoRI.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scDoRI
