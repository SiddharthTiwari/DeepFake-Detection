from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

available_datasets = [
    'ff-deepfake',

]


def load_df(df_path: str,faces_dir: str, dataset: str) -> (pd.DataFrame, str):
    if dataset.startswith('ff-'):
        df = pd.read_pickle(df_path)
        root = faces_dir
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return df, root


def get_split_df(df: pd.DataFrame, dataset: str, split: str) -> pd.DataFrame:
    if dataset.startswith('ff-'):
        # Save random state
        st0 = np.random.get_state()
        # Set seed for this selection only
        np.random.seed(41)
        # Split on original videos
        random_videos = np.random.permutation(
            df['video'].unique())
        train_orig = random_videos[:280]
        val_orig = random_videos[280:280 + 60]
        test_orig = random_videos[280 + 60:]
        if split == 'train':
            split_df = df[df['video'].isin(train_orig)]
        elif split == 'val':
            split_df = df[df['video'].isin(val_orig)]
        elif split == 'test':
            split_df = df[df['video'].isin(test_orig)]
        else:
            raise NotImplementedError('Unknown split: {}'.format(split))
            
        # Restore random state
        np.random.set_state(st0)
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(dataset))
    return split_df


def make_splits( faces_df: str, faces_dir: str, dbs: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple[pd.DataFrame, str]]]:
    """
    Make split and return Dataframe and root
    """
    split_dict = {}
    full_dfs = {}
    for split_name, split_dbs in dbs.items():
        split_dict[split_name] = dict()
        for split_db in split_dbs:
            if split_db not in full_dfs:
                full_dfs[split_db] = load_df(faces_df, faces_dir, split_db)
            full_df, root = full_dfs[split_db]
            split_df = get_split_df(df=full_df, dataset=split_db, split=split_name)
            split_dict[split_name][split_db] = (split_df, root)

    return split_dict
