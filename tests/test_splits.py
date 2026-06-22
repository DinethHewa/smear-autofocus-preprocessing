import pandas as pd

from autofocus.data.splits import assign_nested_splits, assert_no_group_overlap, assert_test_never_in_inner


def test_assign_nested_splits_group_disjoint():
    df = pd.DataFrame({
        'dataset': ['a'] * 8,
        'stack_id': [f's{i}' for i in range(8)],
        'group_id': ['g0', 'g0', 'g1', 'g1', 'g2', 'g2', 'g3', 'g3'],
        'path': [f'p{i}' for i in range(8)],
        'stack_index': [0] * 8,
    })
    out = assign_nested_splits(df, group_col='group_id', outer_test_size=0.33, inner_val_size=0.33, seed=123)
    assert_no_group_overlap(out, 'group_id', 'outer_split')
    trainval = out[out['outer_split'] == 'trainval']
    assert_no_group_overlap(trainval, 'group_id', 'inner_split')
    assert_test_never_in_inner(out)
