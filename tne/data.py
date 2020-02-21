import datetime
import dill
import tqdm.auto
import pathlib
import zipfile
import numpy as np
import pandas as pd


def parse_interaction_events(data_path, first_day_date, from_date_incl, to_date_excl, num_timesteps=48, bidirectional=True):
    dfs = []

    for filename in tqdm.auto.tqdm(sorted(list(pathlib.Path(data_path).glob('**/*.zip')))):
        index = int(filename.name.split('.')[0].split('_')[1])
        day, _ = divmod(index, num_timesteps)

        date = first_day_date + datetime.timedelta(days=day)

        if (date < from_date_incl) or (date >= to_date_excl):
            continue

        day = (date - from_date_incl).days

        with zipfile.ZipFile(filename) as resultsfile:
            with resultsfile.open('results.dill') as results:
                t = dill.loads(results.read())
                dfs.append(pd.DataFrame(t, columns=['timestamp', 'bee_id_0', 'bee_id_1']))

    df = pd.concat(dfs)
    df.drop_duplicates(inplace=True)

    if bidirectional:
        df_swapped = df.copy()
        df_swapped.loc[:, ['bee_id_0','bee_id_1']] = df_swapped.loc[:, ['bee_id_1','bee_id_0']].values
        df = pd.concat((df, df_swapped))
        df.sort_values('timestamp', inplace=True)
        df.reset_index(inplace=True, drop=True)

    return df


def load_alive_df(alive_path):
    alive_df = pd.read_csv(alive_path, parse_dates=['annotated_tagged_date', 'inferred_death_date'])

    return alive_df


def filter_alive_bees(df, alive_path, date):
    alive_df = load_alive_df(alive_path)

    daily_alive = ((alive_df.annotated_tagged_date <= date) & (alive_df.inferred_death_date > date)).values
    bee_ids = alive_df[daily_alive].bee_id

    keep_idx = df.bee_id_0.isin(bee_ids) & df.bee_id_1.isin(bee_ids)
    df = df[keep_idx].copy()

    df.sort_values('timestamp', inplace=True)

    return df, sorted(bee_ids)


def map_bee_ids(df, bee_ids):
    id_to_idx = dict([(bee_id, idx) for (idx, bee_id) in enumerate(bee_ids)])
    idx_to_id = dict([(idx, bee_id) for (idx, bee_id) in enumerate(bee_ids)])

    df.bee_id_0 = df.bee_id_0.apply(lambda bee_id: id_to_idx[bee_id])
    df.bee_id_1 = df.bee_id_1.apply(lambda bee_id: id_to_idx[bee_id])

    return df, id_to_idx, idx_to_id


def subsample_individuals(df, num_individuals):
    ids, counts = np.unique(df.bee_id_0, return_counts=True)
    p = counts.astype(np.float) / np.sum(counts)
    bee_ids = np.random.choice(ids, size=num_individuals, replace=False, p=p)
    df = df[df.bee_id_0.isin(bee_ids) & df.bee_id_1.isin(bee_ids)]

    return df, bee_ids


def get_ages(date, alive_path, bee_ids):
    alive_df = load_alive_df(alive_path)

    tagged_dates = alive_df[alive_df.bee_id.isin(bee_ids)].sort_values('bee_id').annotated_tagged_date
    ages_in_days = (date - tagged_dates).apply(lambda dt: dt.days).values

    return ages_in_days
