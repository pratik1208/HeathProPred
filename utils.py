def encode(data_frame,encoder):
    df_copy = data_frame.copy()
    for col in data_frame.columns:
        if col in encoder:
            df_copy[col] = df_copy[col].apply(
                lambda x: encoder[col][x]
                if x and (x is not np.nan) and (x in encoder[col]) else 0
            )
    return df_copy