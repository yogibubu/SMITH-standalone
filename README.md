# Standalone SMITH / SONIC

This package is the self-contained publication distribution of SMITH.  It
builds a frozen SONIC internal-coordinate contract without installing or
accessing any other repository.  The small topology, primitive-coordinate and
Gaussian-export libraries required by the release are included in the source
distribution, together with the revision-pinned ORACLE point-group kernel.

SMITH uses a provider-neutral input boundary.  It can start from Cartesian
geometry plus a supplied topology, from a supplied redundant primitive/Wilson-B
contract, or from a complete frozen molecular state.  With plain Cartesian
input, a bundled perception frontend constructs the topology, point-group
operations, atom permutations and ordinary primitives needed by SONIC.

The permanent boundary is:

- an input provider supplies Cartesian geometry and may also supply topology,
  redundant primitives, symmetry, fragments, interaction centres, or
  continuous descriptors;
- SMITH constructs and validates SONIC, writes its analytic Wilson B matrix,
  symmetry/rank diagnostics, human-readable report, and optional Gaussian 16
  serialization;
- optimization, scans, finite internal-to-Cartesian realization, force fields,
  Hessian transport, and higher derivatives are application responsibilities.

In particular, SMITH does not construct or serialize B-prime.  A program that
transforms Hessians away from a stationary point can evaluate B-prime on demand
from the frozen coordinate definitions.

In the complete MATRIX suite, ORACLE is the authoritative provider of symmetry,
topology, primitives and B; SMITH consumes these sections without reperception.
The standalone package embeds a revision-pinned subset of that implementation
only to remove a runtime dependency.  LINK owns all finite
internal-to-Cartesian realization, and ARCHITECT owns B-prime and nonstationary
Hessian transformation. MORPHEUS and SENTINEL do not perform their own B
inversion; they request geometry services from LINK.

Four input profiles are available: `FROZEN_STATE`,
`STANDALONE_TOPOLOGY`, `STANDALONE_PRIMITIVES`, and
`STANDALONE_MINIMAL`.  The last profile is sufficient for ordinary inputs and
the paper examples and includes ordinary point-group perception.  Advanced
continuous descriptors, nondefault quasi-symmetry decisions, fragments or
interaction centers can be supplied in a complete frozen state.

## Install from GitHub

Python 3.11 or newer and `git` are required.

```bash
export SMITH_ENV=/path/to/your/smith-venv
python -m venv "$SMITH_ENV"
source "$SMITH_ENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install \
  "smith-sonic @ git+https://github.com/yogibubu/Smith.git@v0.1.0rc7#subdirectory=standalone"
```

The environment directory is selected by the installer.  The package contains
no absolute path to the developer's machine.
The immutable tag is deliberate; replacing it by `main` would make the
manuscript environment depend on later, unvalidated changes.

For a local checkout:

```bash
python -m pip install ./standalone
```

## Run

The examples are installed inside the package, so they can be run even without
a source checkout:

```bash
smith-sonic example water water.xyzin
smith-sonic inspect water.xyzin
smith-sonic example norbornane norbornane.xyzin
smith-sonic example formic-acid-water formic-acid-water.xyzin
smith-sonic example water-dimer water-dimer.xyzin
smith-sonic example benzene-water benzene-water.xyzin
smith-sonic example saccharin saccharin.xyzin
smith-sonic example eta3-allyl-palladium eta3-allyl-palladium.xyzin
```

The same input files are visible under `examples/` in the GitHub repository and
can be passed explicitly to `smith-sonic build`.

To require a complete externally validated state, make the boundary explicit:

```bash
smith-sonic build molecule.xyzin molecule.sonic.xyzin --require-frozen-state
```

Every output receives a `#SMITH_PROVENANCE` section recording whether SMITH
consumed a complete state, supplied topology, supplied primitives, or its
minimal Cartesian frontend.  The package deliberately limits its surface to
SONIC contract construction and inspection.  Each build
also writes a `.smith.out` coordinate report and a `.g16.gjf` Gaussian 16
Rev. C.01 input.  Gaussian 16 Rev. C.01 is used because `ReadAllGIC` provides a general independent
interpreter for SONIC expressions.  SONIC serialization is one particular use
among the many coordinate definitions supported by the GIC language.  The G16
profile is the default, and
non-totally symmetric coordinates are written as `Frozen`.  Because Gaussian
16 Rev. C.01 has no native SONIC out-of-plane primitive and cannot safely represent every
special or multi-periodic composite coordinate, the exporter translates
out-of-plane rows to improper dihedrals and emits supported component
coordinates where necessary.  The native SONIC contract and human report are
never altered by this terminal compatibility translation.

With `--require-frozen-state`, the required production boundary is
`VALIDATION`, `TOPOLOGY`, `SYNTHONS`, `SYMMETRY`, and `PRIMITIVES`. The last
section freezes the primitive ordering, reference values and Wilson-B
fingerprint that SMITH consumes before constructing SONIC.  Without this flag,
a file carrying only `TOPOLOGY` causes SMITH to generate the ordinary redundant
primitives, while a file carrying `PRIMITIVES` uses those rows directly after
geometry and topology-consistency checks.

The formic-acid–water, water-dimer, and benzene–water examples exercise all six
intermolecular fragment coordinates in hydrogen-bonded and aromatic–polar
complexes.  The eta3 allyl–palladium example consumes an explicitly supplied
interaction centre; its idealized geometry is an interface test, not a computed
chemical benchmark.  See [`MANUAL.md`](MANUAL.md) for the complete
installation, input, example, output, and troubleshooting guide.

## Validation status

This submission candidate has completed external collaborator validation.  Its
publication gate also verifies clean installations with Python 3.11 and 3.13,
all four input profiles, Gaussian 16 export, and consumption of the frozen
SONIC contract by LINK and MORPHEUS.  Changes after the submission freeze must
be released as a later version.

The downstream probe is a development integration gate, not a runtime
dependency of the standalone package:

```bash
MATRIX_ROOT=/absolute/path/to/MATRIX SMITH_REQUIRE_DOWNSTREAM=1 \
  python -m pytest standalone/tests/test_downstream_consumers.py
```
