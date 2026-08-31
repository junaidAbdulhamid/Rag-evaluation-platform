"""Experiments: one ExperimentConfig -> run_experiment(config) -> ExperimentResult.

This package is the integration layer. It owns no evaluation logic of its own - it
wires the Phase 1-6 pieces together according to a config, times each stage, meters
tokens, estimates cost, and produces one saved result object.
"""
