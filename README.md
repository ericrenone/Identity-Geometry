# Identity-Geometry

**Fisher-Rao information geometry + rational inattention novelty scoring for LLM outputs.**

Lightweight, research-grade Python library for computing **novelty and information density** in language models. Provides reproducible **diagonal Fisher trace**, **KL divergence**, and a **novelty functional** that combines both measures.

---

## Installation

```bash
pip install git+https://github.com/yourname/identity-geometry.git


This project provides a minimal, production-ready way to measure informational novelty in Large Language Model (LLM) outputs.

The question it answers is simple:

How much new information does this text contain from the model’s point of view?

Instead of relying on embeddings, sampling, or human-style creativity metrics, this approach measures novelty using signals internal to the model itself. The result is a deterministic, auditable score that reflects how strongly the model reacts to a given input.

What “Novelty” Means Here

In this context, novelty does not mean creativity, style, or semantic distance.

Novelty means:

The model makes a confident, non-generic prediction

The input meaningfully activates the model’s parameters

The signal is not inflated by long or repetitive text

In other words, the text contains information that is non-trivial for the model.

How It Works (Conceptually)

The novelty score combines three internal signals:

Prediction Confidence
Measures how concentrated the model’s next-token prediction is. Generic text produces flat, uniform predictions. Informative text produces sharper ones.

Parameter Sensitivity
Measures how strongly the input affects the model’s internal parameters. Inputs the model has truly “learned from” tend to activate parameters more strongly.

Length Normalization
Penalizes long or repetitive inputs so novelty reflects information density, not size.

These components are combined into a single scalar score.

How to Interpret the Score

High score
The model is confident and strongly affected by the input. The text is genuinely informative relative to the model.

Medium score
The text is mostly predictable but contains some meaningful signal.

Low score
The text is generic, boilerplate, repetitive, or well-memorized by the model.

Why This Is Useful

This metric is useful when you need:

Deterministic novelty scoring (no randomness)

Model-aware evaluation instead of surface heuristics

Auditable signals suitable for production or regulated environments

Common use cases include:

Prompt quality evaluation

Dataset curation and deduplication

Memorization or leakage detection

Redundancy filtering

Agent self-evaluation and introspection

Design Principles

Deterministic: no sampling, no randomness

Model-internal: uses the model’s own signals

Scalable: supports layer-filtered computation

Auditable: simple, interpretable components

Minimal: single-file implementation
