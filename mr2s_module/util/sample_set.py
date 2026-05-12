from dimod import SampleSet


def empty_binary_sample_set() -> SampleSet:
  """Return an empty BINARY SampleSet for deterministic shortcut solutions."""
  return SampleSet.from_samples([], vartype="BINARY", energy=[])
