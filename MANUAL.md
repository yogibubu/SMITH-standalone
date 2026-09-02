# Standalone SMITH / SONIC manual

## 1. Scope

Standalone SMITH converts a Cartesian molecular description and its perceived
molecular state into a rank-complete SONIC internal-coordinate contract.  It is
the self-contained, reproducible distribution associated with the SMITH
manuscript.

The standalone input boundary is provider-neutral: SMITH accepts supplied
topology, supplied redundant primitive/Wilson-B definitions, a complete frozen
state, or plain Cartesian geometry through a bundled perception frontend.
The bundled frontend is a revision-pinned subset of ORACLE and constructs
ordinary topology, molecular point-group operations, atom permutations and
primitive coordinates.  It is included so that standalone SMITH has no runtime
dependency on another repository.

An input provider may supply topology, cycles, symmetry, atom equivalence,
continuous descriptors, interaction centres, and the redundant
primitive/Wilson-B source.  SMITH constructs coordinate candidates, protected
special coordinates, rank reduction, symmetry adaptation, analytic Wilson
rows, and the serialized SONIC contract.  Its boundary ends there.
Optimization, scans, finite internal-to-Cartesian realization, force fields,
Hessian transport and higher derivatives belong to applications consuming the
contract.  In particular, B-prime is not required, constructed, or stored by
SMITH; it is evaluated on demand by a second-order consumer when needed.

The corresponding MATRIX ownership is: ORACLE provides authoritative symmetry,
primitive coordinates and B; SMITH consumes them without reperception and
builds SONIC; LINK realizes internal-coordinate changes in
Cartesian space; ARCHITECT evaluates B-prime and transforms nonstationary
Hessians. MORPHEUS delegates realization to LINK, while SENTINEL proposes only
SONIC points through the LINK protocol.

## 2. Requirements and installation

Use Python 3.11 or newer and Git.  The release candidate contains all required
SMITH sources; no second repository or machine-specific path is needed.

```bash
gh auth login
gh auth setup-git
export SMITH_CHECKOUT=/path/to/your/Smith
export SMITH_ENV=/path/to/your/smith-venv
git clone https://github.com/yogibubu/Smith.git "$SMITH_CHECKOUT"
cd "$SMITH_CHECKOUT"
git switch --detach v0.1.0rc7
python3 -m venv "$SMITH_ENV"
source "$SMITH_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install ./standalone
```

Replace the two `/path/to/your/...` values with directories chosen by the user.
On Windows PowerShell, define the corresponding paths and activate the
environment with `<smith-env>\Scripts\Activate.ps1`.  The detached checkout is
intentional and fixes the complete standalone source snapshot.

## 3. Command summary

```text
smith-sonic --version
smith-sonic example NAME [OUTPUT]
smith-sonic build INPUT [OUTPUT] [--require-frozen-state]
smith-sonic inspect FILE
```

`example` runs an input installed with the package.  `build` accepts a plain
extended XYZ input, an XYZ carrying `TOPOLOGY` or `PRIMITIVES`, or a complete
frozen `xyzin`.  `inspect` reports the
presence of the frozen state, GIC section, and provenance profile.

Every `build` or `example` command writes three files with the same stem:

- `.xyzin`: frozen SMITH/SONIC contract;
- `.smith.out`: readable coordinate report with rank, families, protected
  rows, values, units, irreps, active/frozen status, and primitive coefficients;
- `.g16.gjf`: Gaussian 16 Rev. C.01 `ReadAllGIC` optimization input.

Gaussian 16 Rev. C.01 is used because its general `ReadAllGIC` language can interpret
SONIC expressions directly and independently.  SONIC serialization is one
particular use of this much broader GIC language.  The profile is enabled by
default, uses Gaussian-compatible improper dihedrals, and marks every
non-totally symmetric coordinate as `Frozen`; totally symmetric coordinates
remain active.  Fragment and interaction-centre helper functions are serialized
as `Inactive` definitions.  Gaussian 16 Rev. C.01 does not support native SONIC
out-of-plane rows or every special/multi-periodic composite expression.  The
`.g16.gjf` writer therefore performs a terminal compatibility translation;
native out-of-plane and special coordinates remain unchanged in `.xyzin` and
in the human-readable report.

For a complete externally prepared and validated state, require the boundary explicitly:

```bash
smith-sonic build molecule.xyzin molecule.sonic.xyzin --require-frozen-state
```

This option refuses an input missing any of `VALIDATION`, `TOPOLOGY`,
`SYNTHONS`, `SYMMETRY`, or `PRIMITIVES`. The latter contains the ordered
redundant coordinates, reference values and Wilson-B fingerprint.

## 4. Input profiles

### Complete frozen state

This is the production interface.  SMITH preserves the molecular perception
and primitive/B sections and adds the SONIC construction.  The output provenance contains
`PERCEPTION_PROFILE FROZEN_STATE`.

### Supplied topology or primitives

An input carrying `TOPOLOGY` records `STANDALONE_TOPOLOGY` and generates the
ordinary redundant primitive set from that graph.  An input carrying
`PRIMITIVES` records `STANDALONE_PRIMITIVES` and validates the supplied
reference values and Wilson-B fingerprint.  If both sections are present,
their covalent bond rows must agree.

### Minimal Cartesian profile

A plain or extended XYZ is sufficient for small reproducibility examples.  The
packaged path performs the embedded topology/symmetry/primitive pass and records
`PERCEPTION_PROFILE STANDALONE_MINIMAL`; the profile name is retained for schema
compatibility.  For disconnected components it also
constructs fragment definitions and six intermolecular translation/orientation
coordinates.  This profile is deliberately limited; advanced continuous
descriptors, nondefault quasi-symmetry choices, externally audited symmetry and
interaction centers must be supplied in the input when they affect the
requested SONIC contract.

Extended-XYZ directives follow the Cartesian block, for example:

```text
#SMITH
fragment_mode = special-coordinates
symmetrize = false
sycart = false
improper_dihedrals = false
```

The available advanced controls include `symmetry_group`, `xh_stretch_policy`,
`local_xh_bonds`, and `local_xh_classes`.  Omit them to use the pinned SMITH
defaults.

## 5. Packaged examples

Run the complete set with:

```bash
smith-sonic example water water.xyzin
smith-sonic example norbornane norbornane.xyzin
smith-sonic example formic-acid-water formic-acid-water.xyzin
smith-sonic example water-dimer water-dimer.xyzin
smith-sonic example benzene-water benzene-water.xyzin
smith-sonic example eta3-allyl-palladium eta3-allyl-palladium.xyzin
```

Expected results are:

| Example | Input profile | Target rank | Purpose |
|---|---:|---:|---|
| water | standalone minimal | 3 | minimal nonlinear molecule |
| norbornane | standalone minimal | 51 | bridged and cyclic topology |
| formic-acid-water | standalone minimal + fragments | 18 | non-covalent two-fragment contract |
| water-dimer | standalone minimal + fragments | 12 | hydrogen-bonded two-fragment contract |
| benzene-water | standalone minimal + fragments | 39 | aromatic-polar two-fragment contract |
| eta3-allyl-palladium | complete frozen state | 24 | protected metal-to-η3-centre coordinate |

Each of the three non-covalent outputs must contain three `FRAG_TRANSLATION`
and three `FRAG_ORIENTATION` primitives.  The η3 probe must retain an `ETA3_CENTER` over
the three allyl carbon atoms and create a protected `CENTER_ATOM_DISTANCE` from
Pd to that centre.

The η3 geometry is an idealized interface test, not an optimized or computed
chemical benchmark.  Its interaction centre is supplied explicitly in a
frozen input fixture.  It tests whether SMITH consumes the correct contract;
it does not claim that the minimal profile can perceive η3 bonding.

The advanced generated artifacts are retained under `standalone/examples/`:

- `formic_acid_water.xyzin`, `.smith.out`, and `.g16.gjf`;
- `water_dimer.smith.xyz` and `benzene_water.smith.xyz`;
- `eta3_allyl_palladium.xyzin`, `.smith.out`, and `.g16.gjf`.

## 6. Reading an output

The most relevant sections are:

- `TOPOLOGY`, `SYNTHONS`, `SYMMETRY`, and `INTERACTION_CENTERS`: perceived
  molecular state consumed by SMITH;
- `GIC`: build options, target dimension, primitive candidates, protected
  special coordinates, and the selected generalized internal coordinates;
- `SMITH_PROVENANCE`: package version, implementation revision, and input and
  fragment profiles.

For a nonlinear system of `N` atoms without disconnected fragments, the usual
target is `3N - 6`.  Always use the serialized `TARGET_RANK` and reported final
rank when special fragment or centre coordinates are present.

## 7. Verification and problem reports

For the complete standalone publication verification, run:

```bash
python -m pip install "./standalone[test]"
python -m pytest standalone/tests/test_examples.py
```

The public gate checks all input profiles, both output formats, the three
non-covalent examples, and the supplied interaction-centre fixture.
Cross-application integration tests remain development checks and are not
installation requirements for standalone SMITH.

The LINK/MORPHEUS integration probe is intentionally separate because those
applications are not standalone dependencies.  Developers can run it against
a compatible clean MATRIX checkout with:

```bash
MATRIX_ROOT=/absolute/path/to/MATRIX SMITH_REQUIRE_DOWNSTREAM=1 \
  python -m pytest standalone/tests/test_downstream_consumers.py
```

When reporting a problem, include the operating system and architecture,
Python version, exact command, complete terminal output, input file, and
generated `xyzin` if one was produced.  Do not include credentials or private
tokens.  Open an issue in the repository or send the report directly to the
project owner, and identify the frozen release version being tested.

Execution of Gaussian jobs and all optimization or finite-displacement
workflows remain outside this package.  SMITH generates the Gaussian 16 input
but does not launch the external executable.
