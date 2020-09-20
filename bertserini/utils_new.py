def get_best_answer(candidates, weight=0.5):
    for ans in candidates:
        ans.aggregate_score(weight)
    return candidates.sorted(key=lambda x: x.total_score, reverse=True)[0]
