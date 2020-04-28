import sys
from typing import Optional, List, Dict, Tuple

import logging

from vizseq._data import (
    PathOrPathsOrDictOfStrList,
    VizSeqDataSources,
    VizSeqTableExporter,
)
from vizseq.scorers import get_scorer_ids, get_scorer, get_scorer_name

__all__ = ["get_scores", "scores_to_latex"]

logger = logging.getLogger("scores")
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


def get_scores(
    sources: PathOrPathsOrDictOfStrList,
    references: PathOrPathsOrDictOfStrList,
    model_to_hypotheses: PathOrPathsOrDictOfStrList,
    metrics: List[str],
    tags: Optional[PathOrPathsOrDictOfStrList] = None,
    verbose: bool = False,
    problem: str = None,
) -> Tuple[Dict, Dict]:
    # Copyright (c) Facebook, Inc. and its affiliates.
    # The code in this function is licensed under the MIT license.
    _srcs = VizSeqDataSources(sources)
    _refs = VizSeqDataSources(references)
    _hypos = VizSeqDataSources(model_to_hypotheses)
    _tags, tag_set = None, []
    if tags is not None:
        _tags = VizSeqDataSources(tags, text_merged=True)
        tag_set = sorted(_tags.unique())
        _tags = _tags.text
    models = _hypos.names
    all_metrics = get_scorer_ids()
    _metrics = []
    for s in metrics:
        if s in all_metrics:
            _metrics.append(s)
        else:
            logger.warning(f'"{s}" is not a valid metric.')

    def scorer_kwargs(s):
        kwargs = {"corpus_level": True, "sent_level": False, "verbose": verbose}
        if s in (
            "kendall_task_ranking",
            "req_cov",
            "essential_req_cov",
            "achievement",
            "granularity",
        ):
            # ProcGenScorer's
            kwargs["extra_args"] = {"problem": problem}
        return kwargs

    scores = {
        s: {
            m: get_scorer(s)(**scorer_kwargs(s)).score(
                _hypos.data[i].text, _refs.text, tags=_tags, sources=_srcs.text
            )
            for i, m in enumerate(models)
        }
        for s in _metrics
    }

    corpus_scores = {
        s: {m: scores[s][m].corpus_score for m in models} for s in _metrics
    }
    group_scores = {
        s: {t: {m: scores[s][m].group_scores[t] for m in models} for t in tag_set}
        for s in _metrics
    }

    return corpus_scores, group_scores


def scores_to_latex(scores: Dict) -> str:
    return VizSeqTableExporter.to_latex(
        {get_scorer_name(s): scores for s, scores in scores.items()}
    )
