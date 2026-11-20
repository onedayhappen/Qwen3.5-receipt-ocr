"""最大权重一对一商品行匹配；重复预测不能复用同一真值。"""

from difflib import SequenceMatcher

from .normalization import normalize_value

ITEM_FIELDS = ("name", "quantity", "price")


def _known(value):
    return value is not None and value != ""


def _row(row):
    return {field: normalize_value(field, row.get(field)) for field in ITEM_FIELDS}


def _pair(pred, gold):
    available = [field for field in ITEM_FIELDS if _known(gold[field])]
    scores = {field: int(pred[field] == gold[field]) for field in available}
    if not available:
        return 0.0, 0.0, scores
    if _known(gold["name"]):
        if not _known(pred["name"]):
            return 0.0, 0.0, scores
        similarity = SequenceMatcher(None, str(pred["name"]).casefold(),
                                     str(gold["name"]).casefold(), autojunk=False).ratio()
        if similarity < 0.5:
            return 0.0, 0.0, scores
        weight = 3.0 * similarity + sum(scores.values())
    else:
        weight = float(sum(scores.values()))
        if weight == 0:
            return 0.0, 0.0, scores
    return weight, sum(scores.values()) / len(available), scores


def _maximum_assignment(weights):
    """方阵 Hungarian，补零虚拟行列；返回预测行到真值列的分配。"""
    rows = len(weights)
    columns = len(weights[0]) if rows else 0
    size = max(rows, columns)
    if size == 0:
        return []
    costs = [[-weights[i][j] if i < rows and j < columns else 0.0
              for j in range(size)] for i in range(size)]
    u, v, p, way = ([0.0] * (size + 1), [0.0] * (size + 1),
                    [0] * (size + 1), [0] * (size + 1))
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = float("inf"), 0
            for j in range(1, size + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, size + 1)
            if 0 < p[j] <= rows and j <= columns]


def _rates(correct, pred_count, gold_count):
    precision = correct / pred_count if pred_count else 0.0
    recall = correct / gold_count if gold_count else 0.0
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision_denominator": pred_count, "recall_denominator": gold_count,
        "correct": correct,
    }


def match_items(pred_items, gold_items):
    if pred_items is None:
        pred_items = []
    if not isinstance(pred_items, list) or not isinstance(gold_items, list):
        raise TypeError("商品明细必须是 list，未知真值不能当成空列表")
    if any(not isinstance(row, dict) for row in pred_items + gold_items):
        raise TypeError("商品行必须是 dict")
    predictions, truths = list(map(_row, pred_items)), list(map(_row, gold_items))
    pairs = [[_pair(pred, gold) for gold in truths] for pred in predictions]
    weights = [[pair[0] for pair in row] for row in pairs]
    assignments = _maximum_assignment(weights)
    matched = []
    used_pred, used_gold = set(), set()
    for pred_index, gold_index in assignments:
        weight, quality, scores = pairs[pred_index][gold_index]
        if weight <= 0:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matched.append({
            "pred_index": pred_index, "gold_index": gold_index,
            "score": quality, "weight": weight, "field_scores": scores,
        })
    matched.sort(key=lambda pair: pair["pred_index"])
    extra = [index for index in range(len(predictions)) if index not in used_pred]
    missed = [index for index in range(len(truths)) if index not in used_gold]
    correct = sum(pair["score"] for pair in matched)
    result = _rates(correct, len(predictions), len(truths))
    field_scores = {}
    for field in ITEM_FIELDS:
        gold_count = sum(_known(row[field]) for row in truths)
        correct_field = sum(pair["field_scores"].get(field, 0) for pair in matched)
        pred_count = sum(field in pair["field_scores"] for pair in matched)
        if gold_count:
            pred_count += len(extra)
        field_scores[field] = _rates(correct_field, pred_count, gold_count)
        field_scores[field]["applicable"] = gold_count > 0
    duplicate_extra = [index for index in extra if any(
        predictions[index] == predictions[other] for other in range(len(predictions))
        if other != index
    )]
    result.update({
        "matched": matched, "extra": extra, "missed": missed,
        "matched_count": len(matched), "extra_count": len(extra), "missed_count": len(missed),
        "pred_count": len(predictions), "gold_count": len(truths),
        "field_scores": field_scores, "duplicate_extra": duplicate_extra,
        "extra_rate": len(extra) / len(predictions) if predictions else 0.0,
        "exact": len(predictions) == len(truths) == len(matched)
                 and all(pair["score"] == 1.0 for pair in matched),
    })
    return result
