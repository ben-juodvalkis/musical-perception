"""
Evaluation harness (ADR-009).

Tiers 0–1 of the eval ladder: synthetic sweeps over the precision layer and
frozen-trace replay of analyze(). Run with:

    python -m musical_perception.evals run --suite tier0,tier1
"""
