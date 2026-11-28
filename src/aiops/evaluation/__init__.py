"""Evaluation: run RCA methods over the blind view and score them the way RCAEval does.

Method side, label-free::

    from aiops.evaluation.runner import Prediction, run_method
    from aiops.evaluation.baselines import AnomalyRanking

Evaluator side, reads the labels::

    from aiops.evaluation.scoring import score_cases, summarize

This package module imports neither, so importing the method side never loads
the labels module.
"""
