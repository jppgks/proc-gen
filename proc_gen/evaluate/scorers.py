# Copyright (c) 2020-2021 Joppe Geluykens
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import re
from collections import OrderedDict
from typing import List, Optional

import numpy as np
import pandas as pd

from proc_gen.data import string_to_requirements
from proc_gen.data.to_example import string_to_tasks
from vizseq.scorers import VizSeqScorer, VizSeqScore, register_scorer


class ScoreComputationError(ValueError):
    pass


#
# # from vizseq.scorers.bert_score import BERTScoreScorer
#
bert_scorer = None


from collections import namedtuple

Req = namedtuple("Req", ["object", "optional"])


def compute_requirement_coverage(
    hypo: str, req_str: str, essential: bool = False, problem: str = None
) -> float:
    # Parse requirements
    # if problem == 'TargetProductAndRequirementsAndTasks':
    #     # HACK: RecipeGPT grammar
    #     reqs = req_str.rstrip(' <end-ingredients> <start-directions>').split(
    #         ' <end-title> <start-ingredients>')[1].rstrip('$').split('$')
    #     reqs = list(map(lambda r: Req(r, False), reqs))
    if problem in (
        "TargetProductAndRequirements_TO_Tasks",
        "TargetProductAndRequirementsAndTasks",
    ):
        tp_and_reqs = string_to_requirements(req_str, parse_tp=True)
        _, reqs = tp_and_reqs[0], tp_and_reqs[1:]
        if not isinstance(reqs, list):
            reqs = [reqs]
    else:
        reqs = string_to_requirements(req_str)
    if essential:
        reqs = list(filter(lambda r: not r.optional, reqs))
    num_total = len(reqs)

    if num_total == 0:
        return 1

    # Check all requirements
    num_covered = 0
    for req in reqs:
        # Split multi-word objects (matching one word suffices)
        words_in_req = set(req.object.lower().split())
        if set(hypo.lower().split()).intersection(words_in_req):
            num_covered += 1

    return num_covered / num_total


@register_scorer("req_cov", "Requirement Coverage")
class RequirementCoverageScorer(VizSeqScorer):
    # TODO: test this function
    def score(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        tags: Optional[List[List[str]]] = None,
        sources: Optional[List[List[str]]] = None,
    ) -> VizSeqScore:
        problem = self.extra_args["problem"]
        # Only relevant if predicting tasks
        assert problem in (
            "TargetProductAndRequirements_TO_Tasks",
            "Requirements_TO_TargetProductAndTasks",
            "TargetProductAndRequirementsAndTasks",
            "RequirementsAndTargetProductAndTasks",
        )

        corpus_score, sent_scores, group_scores = None, None, None

        requirements = sources[0]
        hypotheses = hypothesis

        sent_scores = []
        for req_str, hypo in zip(requirements, hypotheses):
            try:
                score = compute_requirement_coverage(hypo, req_str, problem=problem)
            except ValueError:
                score = 0
            sent_scores.append(score)

        if self.corpus_level:
            corpus_score = np.mean(sent_scores) * 100

        return VizSeqScore.make(
            corpus_score=corpus_score, sent_scores=sent_scores, group_scores={}
        )


@register_scorer("essential_req_cov", "Essential Requirement Coverage")
class EssentialRequirementCoverageScorer(VizSeqScorer):
    # TODO: test this function
    def score(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        tags: Optional[List[List[str]]] = None,
        sources: Optional[List[List[str]]] = None,
    ) -> VizSeqScore:
        problem = self.extra_args["problem"]
        # Only relevant if predicting tasks
        assert problem in (
            "TargetProductAndRequirements_TO_Tasks",
            "Requirements_TO_TargetProductAndTasks",
            "TargetProductAndRequirementsAndTasks",
            "RequirementsAndTargetProductAndTasks",
        )

        corpus_score, sent_scores, group_scores = None, None, None

        requirements = sources[0]
        hypotheses = hypothesis

        sent_scores = []
        for req_str, hypo in zip(requirements, hypotheses):
            try:
                score = compute_requirement_coverage(
                    hypo, req_str, essential=True, problem=problem
                )
            except ValueError:
                score = 0
            sent_scores.append(score)

        if self.corpus_level:
            corpus_score = np.mean(sent_scores) * 100

        return VizSeqScore.make(
            corpus_score=corpus_score, sent_scores=sent_scores, group_scores={}
        )


def _best_match(tasks_gt, t):
    global bert_scorer

    if not bert_scorer:
        import bert_score as bs

        print("Loading BERTScorer")
        bert_scorer = bs.BERTScorer(
            model_type="distilbert-base-uncased-distilled-squad",
            nthreads=1,
            lang="en",
            rescale_with_baseline=True,
        )
        print("Done loading BERTScorer")

    assert t and len(tasks_gt)
    t = t.lower()
    tasks_gt = [task.lower() for task in tasks_gt]
    all_scores = bert_scorer.score(
        list(map(lambda s: s.lower(), tasks_gt)),
        [t.lower() for _ in range(len(tasks_gt))],
        verbose=True,
    )[2].tolist()
    best_match_index = np.argmax(all_scores)

    # Return one-based index
    return best_match_index + 1


def compute_task_order_score(tasks_gt, tasks_pred):
    # TODO: test this function
    if len(tasks_pred) < len(tasks_gt):
        # TODO: Happens around 3/4 of the time, should implement this case.
        raise ScoreComputationError(
            "More predicted tasks than ground truth tasks." "Behavior not implemented."
        )

    # Ranks for ground truth
    ranks_gt = range(1, len(tasks_gt) + 1)

    # Get ranks for pred
    # account for
    #   1) pred tasks being somewhat different from gt
    #       -> best match based on embedding distance (bert score)
    #   2.1) ground truth task comprising multiple predicted tasks
    #       -> Kendall Tau allows ties
    # TODO: not yet accounted for
    #   2.2) predicted task comprising multiple ground truth tasks
    ranks_pred = [_best_match(tasks_gt, t) for t in tasks_pred[: len(ranks_gt)]]
    # assert ranks are not constant
    if len(set(ranks_pred)) == 1:
        raise ScoreComputationError(
            f"Predicted ranks were constant {ranks_pred}. " f"Kendall Tau not defined."
        )

    kendall_tau = (
        pd.DataFrame(np.column_stack([ranks_gt, ranks_pred])).corr("kendall").loc[0, 1]
    )
    if np.isnan(kendall_tau):
        raise ScoreComputationError("Kendall Tau was NaN.")

    return kendall_tau


@register_scorer("kendall_task_ranking", "Kendall τ (task ranking)")
class KendallTaskRankingScorer(VizSeqScorer):
    def score(
        self,
        hypothesis: List[str],
        references: List[List[str]],
        tags: Optional[List[List[str]]] = None,
        sources: Optional[List[List[str]]] = None,
    ) -> VizSeqScore:
        # global bert_scorer
        problem = self.extra_args["problem"]

        # Only relevant if predicting tasks
        assert problem in (
            "TargetProductAndRequirements_TO_Tasks",
            "Requirements_TO_TargetProductAndTasks",
            "TargetProductAndRequirementsAndTasks",
            "RequirementsAndTargetProductAndTasks",
        )

        corpus_score, sent_scores, group_scores = None, None, None

        # requirements = sources[0]
        references = references[0]
        hypotheses = hypothesis

        sent_scores = []
        for ref, hypo in zip(references, hypotheses):
            # if problem == 'TargetProductAndRequirementsAndTasks':
            # # HACK: RecipeGPT grammar
            # tasks_gt = re.split('\. |! ', ref.rstrip(' <end-directions>'))
            # tasks_pred = re.split('\. |! ',
            #                       hypo.rstrip(' <end-directions>'))
            if problem in (
                "Requirements_TO_TargetProductAndTasks",
                "RequirementsAndTargetProductAndTasks",
            ):
                # Requirements_TO_TargetProductAndTasks
                tgt_prod_and_tasks_gt = string_to_tasks(ref, parse_tp=True)
                tasks_gt = tgt_prod_and_tasks_gt[1:]

                try:
                    tgt_prod_and_tasks_pred = string_to_tasks(hypo, parse_tp=True)
                    tasks_pred = tgt_prod_and_tasks_pred[1:]
                except ValueError:
                    continue
            else:
                # TargetProductAndRequirements_TO_Tasks or TargetProductAndRequirementsAndTasks
                tasks_gt = string_to_tasks(ref)
                tasks_pred = string_to_tasks(hypo)

            try:
                score = compute_task_order_score(tasks_gt, tasks_pred)
                sent_scores.append(score)
            except ScoreComputationError:
                continue

        if self.corpus_level:
            print(len(sent_scores))
            corpus_score = np.mean(sent_scores)

        return VizSeqScore.make(
            corpus_score=corpus_score, sent_scores=sent_scores, group_scores={}
        )
