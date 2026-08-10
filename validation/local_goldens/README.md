# Local golden data

This directory documents the expected organization of independent validation
goldens. Large raw traces should normally remain local and are not required to
be committed.

Recommended layout:

```text
validation/local_goldens/
  scalesim/<commit>/
    tensor/
    conv/
  rtl/<commit>/spn/
  actsim/<arch_revision>/
  board/<runtime_revision>/
```

Each experiment directory should preserve:
- source git/revision identifier;
- configuration/architecture file;
- exact command line;
- raw log or trace;
- normalized metric JSON;
- calibration/hold-out designation.

Do not overwrite a golden after changing its source simulator or hardware
configuration. Create a new revision directory instead.

The repository comparator expects normalized external metrics of the form:

```json
{
  "metrics": {
    "tensor.dec2.cycles": 12345,
    "tensor.dec2.act_bytes": 12345,
    "spn.nlspn.cycles": 12345,
    "spn.nlspn.gather_reads": 12345
  }
}
```

See `../../CALIBRATION_PLAN.md` and `../README.md` for the staged gates.
