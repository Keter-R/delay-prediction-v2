import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import random_projection


def word2vec(word_list, dim=3):
    if dim == 0:
        return None
    vectorizer = HashingVectorizer(n_features=dim)
    vec = vectorizer.transform(word_list)
    vec = vec.toarray()
    print(vec.shape)
    return vec


def purify_xlsx(source_path, dest_path, target_cols, target_routes=None):
    dat = pd.read_excel(source_path)
    dat = dat.dropna()
    if 'Line' in dat.columns:
        dat = dat.rename(columns={'Line': 'Route'})
    if 'Min Delay' in dat.columns:
        dat = dat.rename(columns={'Min Delay': 'Delay'})
    if target_routes is not None:
        dat = dat[dat['Route'].astype(str).apply(lambda x: x in target_routes)]
    else:
        # for each row, if Route is not all numeric, then drop this row
        dat = dat[dat['Route'].astype(str).apply(lambda x: x.isnumeric())]
    # --------
    # remove rows that in 'Bound' column is not 'W' or 'E' or 'N' or 'S' or 'B'
    dat = dat[dat['Bound'].apply(lambda x: x in ['W', 'E', 'N', 'S', 'B'])]
    # remove rows that in 'Delay' column is zero
    # dat = dat[dat['Delay'] != 0]
    loc_emb = word2vec(dat['Location'], dim=12)
    # --------
    for col in dat.columns:
        if col not in target_cols:
            dat = dat.drop(columns=[col])
    dat = join_time_into_date(dat, 'Date', 'Time')
    # reset indexes of dat and reserve columns
    cols = dat.columns.tolist()
    dat = dat.reset_index(drop=True)
    # add cols back to dat
    dat = pd.DataFrame(dat, columns=cols)
    if loc_emb is not None:
        # add column name to loc_emb
        loc_emb = pd.DataFrame(loc_emb, columns=[f'Location_{i}' for i in range(loc_emb.shape[1])])
        print(loc_emb)
        print(dat)
        dat = pd.concat([dat, loc_emb], axis='columns')
        print(dat)
    dat.to_excel(dest_path, index=False, engine='xlsxwriter')
    return dat


def join_time_into_date(df, date_col, time_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df[time_col] = pd.to_datetime(df[time_col], format='%H:%M')
    df[date_col] = df[date_col] + pd.to_timedelta(df[time_col].dt.strftime('%H:%M:%S'))
    df = df.drop(columns=[time_col])
    return df
