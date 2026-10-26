"""Outcome-free geometry tests of a representation, plus locality of outcomes in it."""
from .cka import linear_cka
from .dataset_structure import cluster_alignment, dataset_identifiability, dataset_variance_fraction
from .locality import neighbour_analysis, pair_analysis
from .spectrum import spectrum_stats

__all__ = ["linear_cka", "cluster_alignment", "dataset_identifiability", "dataset_variance_fraction",
           "neighbour_analysis", "pair_analysis", "spectrum_stats"]
