# License and dependency notices

Original trainer code in this repository is **GPL-3.0-only**. The complete GNU GPL version 3 is in LICENSE. SPDX headers identify copied and newly written code under that license. The project does not offer an “or later” option.

Dependencies are installed by the user; their code is not vendored into this repository. Each dependency retains its own license and notices:

| Dependency | Version | License | Upstream |
| --- | --- | --- | --- |
| Pokémon Showdown | 0.11.11 | MIT | [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) |
| JAX and JAXlib | 0.12.0.1 | Apache-2.0 | [jax-ml/jax](https://github.com/jax-ml/jax) |
| NumPy | 2.4.0 | BSD-3-Clause, with bundled-code notices | [numpy/numpy](https://github.com/numpy/numpy) |
| SciPy | 1.18.1 | BSD-3-Clause, with bundled-code notices | [scipy/scipy](https://github.com/scipy/scipy) |
| ml_dtypes | 0.6.0 | Apache-2.0 | [jax-ml/ml_dtypes](https://github.com/jax-ml/ml_dtypes) |
| opt_einsum | 3.4.0 | MIT | [dgasmith/opt_einsum](https://github.com/dgasmith/opt_einsum) |
| setuptools | 80.9.0 | MIT | [pypa/setuptools](https://github.com/pypa/setuptools) |
| tsx | 4.23.15 | MIT | [privatenumber/tsx](https://github.com/privatenumber/tsx) |
| TypeScript | 5.9.3 | Apache-2.0 | [microsoft/TypeScript](https://github.com/microsoft/TypeScript) |
| Node type definitions | 24.19.0 | MIT | [DefinitelyTyped](https://github.com/DefinitelyTyped/DefinitelyTyped) |

The npm lockfile records transitive package versions, integrity hashes and declared licenses; installed packages contain their full notices. Showdown includes optional server/database dependencies, which these battle actors do not load. The Python requirements lock the CPU dependencies; optional CUDA wheels carry their own NVIDIA/JAX package terms and notices.

Pokémon names, mechanics and other game content supplied by Showdown remain the property of their respective owners. This project is unaffiliated with Pokémon, Nintendo, Game Freak or The Pokémon Company.

**No Foul Play code is included.** No external policy implementation, search engine, trained weight, checkpoint, team pool or replay dataset is included. Random and heuristic baselines are extracted from the original trainer repository's own baseline functions.

No dataset is fetched by the tooling. Teams are supplied locally by the user and are not redistributed here. If a user downloads an external dataset (including Metamon team sets), they must inspect that dataset's actual license: this repository makes no claim that those sets have a license or that a code license covers them. No dataset has been selected, fetched or relicensed for this release.
