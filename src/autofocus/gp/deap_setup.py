from __future__ import annotations

from deap import base, creator


def get_creator() -> None:
    if not hasattr(creator, 'FitnessMin'):
        creator.create('FitnessMin', base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'Individual'):
        creator.create('Individual', list, fitness=creator.FitnessMin)
