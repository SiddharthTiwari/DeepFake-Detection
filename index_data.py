import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from isplutils.utils import extract_meta_av, extract_meta_cv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, help='Source dir',
                        default='dataset/videos')
    parser.add_argument('--videodataset', type=Path, default='data/df_videos.pkl',
                        help='Path to save the videos DataFrame')

    args = parser.parse_args()

    ## Parameters parsing
    source_dir: Path = args.source
    videodataset_path: Path = args.videodataset

    # Create ouput folder (if doesn't exist)
    videodataset_path.parent.mkdir(parents=True, exist_ok=True)

    ## DataFrame
    if videodataset_path.exists():
        print('Loading video DataFrame')
        df_videos = pd.read_pickle(videodataset_path)
    else:
        print('Creating video DataFrame')

        ff_videos = Path(source_dir).rglob('*.mp4')
        df_videos = pd.DataFrame(
            {'path': [f.relative_to(source_dir) for f in ff_videos ]})
        
        df_videos['height'] = df_videos['width'] = df_videos['frames'] = np.zeros(len(df_videos), dtype=np.uint16)
        with Pool() as p:
            meta = p.map(extract_meta_av, df_videos['path'].map(lambda x: str(source_dir.joinpath(x))))
        meta = np.stack(meta)
        df_videos.loc[:, ['height', 'width', 'frames']] = meta
        
        # Fix for videos that av cannot decode properly
        for idx, record in df_videos[df_videos['frames'] == 0].iterrows():
            meta = extract_meta_cv(str(source_dir.joinpath(record['path'])))
            df_videos.loc[idx, ['height', 'width', 'frames']] = meta
            
        df_videos['class'] = df_videos['path'].map(lambda x: x.parts[0]).astype('category')
        df_videos['label'] = df_videos['class'].map(
            lambda x: True if x == 'fake' else False)  # True is FAKE, False is REAL
        
        print('Saving video DataFrame to {}'.format(videodataset_path))
        df_videos.to_pickle(str(videodataset_path))

    print('Real videos: {:d}'.format(sum(df_videos['label'] == 0)))
    print('Fake videos: {:d}'.format(sum(df_videos['label'] == 1)))


if __name__ == '__main__':
    main()
