import pandas as pd


def purify_xlsx(source_path, dest_path, target_cols):
    dat = pd.read_excel(source_path)
    if 'Line' in dat.columns:
        dat = dat.rename(columns={'Line': 'Route'})
    if 'Min Delay' in dat.columns:
        dat = dat.rename(columns={'Min Delay': 'Delay'})
    for col in dat.columns:
        if col not in target_cols:
            dat = dat.drop(columns=[col])
    dat = dat.dropna()
    # for each row, if Route is not all numeric, then drop this row
    dat = dat[dat['Route'].astype(str).apply(lambda x: x.isnumeric())]
    dat = join_time_into_date(dat, 'Date', 'Time')
    dat.to_excel(dest_path, index=False)
    return dat


def join_time_into_date(df, date_col, time_col):
    df[date_col] = pd.to_datetime(df[date_col])
    df[time_col] = pd.to_datetime(df[time_col], format='%H:%M')
    df[date_col] = df[date_col] + pd.to_timedelta(df[time_col].dt.strftime('%H:%M:%S'))
    df = df.drop(columns=[time_col])
    return df
