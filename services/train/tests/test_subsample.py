from agritune_train.sft import subsample


def test_subsample_is_stratified_seeded_and_keeps_every_group():
    rows = [{"db_id": f"db{i % 10}", "n": i} for i in range(1000)] + [{"db_id": "tiny", "n": -1}]
    part = subsample(rows, 0.3, seed=1, key="db_id")
    assert 290 <= len(part) <= 312
    assert {r["db_id"] for r in part} == {r["db_id"] for r in rows}  # the 1-row group survives
    assert part == subsample(rows, 0.3, seed=1, key="db_id")
    assert subsample(rows, 1.0, seed=1) is rows
    assert len(subsample(rows, 0.5, seed=1)) == round(len(rows) * 0.5)
