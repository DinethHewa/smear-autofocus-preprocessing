from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

import pandas as pd
from deap import base, tools, creator

from .deap_setup import get_creator
from .genome import random_genome, genome_to_pipeline, filter_ops_by_availability
from ..eval.evaluate import evaluate_pipeline
from ..preprocess.pipeline import Pipeline, PipelineStep
from ..preprocess.param_spaces import PARAM_SPACES


@dataclass
class GPConfig:
    generations: int
    population_size: int
    elite_size: int
    mutation_rate: float
    crossover_rate: float
    cache: bool = True


class GPOptimizer:
    def __init__(self, categories: List[str], ops_by_category: Dict[str, List[str]], seed: int, config: GPConfig,
                 focus_cfg: Dict[str, float], metrics_cfg: Dict[str, float],
                 evaluation_cfg: Dict[str, float] | None = None,
                 tuning_cfg: Dict[str, float] | None = None,
                 params_lookup: Dict[str, Dict[str, float]] | None = None,
                 strict_params: bool = False) -> None:
        self.categories = categories
        self.ops_by_category = filter_ops_by_availability(ops_by_category)
        self.rng = random.Random(seed)
        self.config = config
        self.focus_cfg = focus_cfg
        self.metrics_cfg = metrics_cfg
        self.evaluation_cfg = evaluation_cfg or {}
        self.tuning_cfg = tuning_cfg or {}
        self.params_lookup = params_lookup or {}
        self.strict_params = strict_params
        self.cache: Dict[Tuple[str, ...], float] = {}
        self.stack_cache: Dict[Tuple[str, str, int | None], Dict[str, float]] = {}
        self.cache_hits = 0
        self.evaluations = 0

        get_creator()
        self.toolbox = base.Toolbox()
        self.toolbox.register('individual', self._init_individual)
        self.toolbox.register('population', tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register('mate', tools.cxOnePoint)
        self.toolbox.register('mutate', self._mutate)
        self.toolbox.register('select', tools.selTournament, tournsize=3)

    def _init_individual(self):
        genome = random_genome(self.rng, self.categories, self.ops_by_category)
        return creator.Individual(genome)

    def _mutate(self, individual):
        for i, category in enumerate(self.categories):
            if self.rng.random() < self.config.mutation_rate:
                individual[i] = self.rng.choice(self.ops_by_category[category])
        return (individual,)

    def _params_for_op(self, op_name: str) -> Dict[str, float]:
        if op_name in self.params_lookup:
            params = self.params_lookup[op_name]
            return params if isinstance(params, dict) else {}
        if self.strict_params and op_name in PARAM_SPACES:
            raise ValueError(f"Missing tuned parameters for operator: {op_name}")
        return {}

    def _apply_tuned_params(self, pipeline: Pipeline) -> Pipeline:
        steps = []
        for step in pipeline.steps:
            params = dict(step.params or {})
            tuned = self._params_for_op(step.name)
            if tuned:
                params.update(tuned)
            steps.append(PipelineStep(
                name=step.name,
                enabled=step.enabled,
                params=params,
                category=step.category,
            ))
        return Pipeline(steps=steps)

    def _evaluate(self, individual, manifest_df: pd.DataFrame, artifacts_dir):
        key = tuple(individual)
        if self.config.cache and key in self.cache:
            self.cache_hits += 1
            return (self.cache[key],)
        pipeline = genome_to_pipeline(individual, self.categories)
        if self.params_lookup:
            pipeline = self._apply_tuned_params(pipeline)
        summary, _ = evaluate_pipeline(
            manifest_df,
            pipeline,
            focus_cfg=self.focus_cfg,
            metrics_cfg=self.metrics_cfg,
            outer_split=['trainval'],
            inner_split=['train', 'val'],
            artifacts_dir=artifacts_dir,
            run_tag='gp_eval',
            failure_penalty=1e6,
            cache=self.stack_cache,
            pipeline_fingerprint=pipeline.fingerprint(),
            evaluation_cfg=self.evaluation_cfg,
            tuning_cfg=self.tuning_cfg,
        )
        score = float(summary['generalization_score'])
        self.evaluations += 1
        if self.config.cache:
            self.cache[key] = score
        return (score,)

    def build_pipeline(self, genome: List[str]) -> Pipeline:
        pipeline = genome_to_pipeline(genome, self.categories)
        if self.params_lookup:
            pipeline = self._apply_tuned_params(pipeline)
        return pipeline

    def run(self, manifest_df: pd.DataFrame, artifacts_dir, progress_cb=None) -> Tuple[List[str], float]:
        pop = self.toolbox.population(n=self.config.population_size)
        hof = tools.HallOfFame(1)

        prev_cache_hits = 0
        prev_evals = 0

        for gen_idx in range(self.config.generations):
            fitnesses = [self._evaluate(ind, manifest_df, artifacts_dir) for ind in pop]
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            hof.update(pop)

            scores = [fit[0] for fit in fitnesses]
            if scores:
                mean_score = float(pd.Series(scores).mean())
                std_score = float(pd.Series(scores).std(ddof=0))
                best_score = float(min(scores))
            else:
                mean_score = float("nan")
                std_score = float("nan")
                best_score = float("nan")

            best_ind = tools.selBest(pop, 1)[0] if pop else None
            best_genome = list(best_ind) if best_ind is not None else []

            gen_cache_hits = self.cache_hits - prev_cache_hits
            gen_evals = self.evaluations - prev_evals
            prev_cache_hits = self.cache_hits
            prev_evals = self.evaluations

            if progress_cb is not None:
                progress_cb({
                    "generation": gen_idx + 1,
                    "best_score": best_score,
                    "mean_score": mean_score,
                    "std_score": std_score,
                    "evaluated": gen_evals,
                    "cache_hits": gen_cache_hits,
                    "best_genome": best_genome,
                })

            offspring = tools.selBest(pop, self.config.elite_size)
            offspring += self.toolbox.select(pop, len(pop) - self.config.elite_size)
            offspring = list(map(self.toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if self.rng.random() < self.config.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if self.rng.random() < self.config.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            pop = offspring

        best = hof[0]
        best_score = best.fitness.values[0]
        return list(best), float(best_score)
