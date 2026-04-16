def run_feature_engineering(path):
    import pandas as pd
    
    df = pd.read_csv(path)
    
    # Check rows per timestamp
    counts = df.groupby("time").size()
    
    print(counts.value_counts())
    
    features = [
        "PRN",
        "Carrier_Doppler_hz",
        "Pseudorange_m",
        "RX_time",
        "TOW",
        "Carrier_phase",
        "EC", "LC", "PC",
        "PIP", "PQP",
        "TCD",
        "CN0"
    ]
    
    def reshape_timestamp(group):
        # group: 8 rows for one timestamp
        
        # sort by channel to maintain consistency
        group = group.sort_values("channel")
        
        # extract feature values
        values = group[features].values  # shape: (8, F)
        
        # flatten → (8*F,)
        flat = values.flatten()
        
        return pd.Series(flat)
    
    
    df_agg = df.groupby("time").apply(reshape_timestamp)
    df_agg.reset_index(inplace=True)
    
    
    new_cols = []
    
    for ch in range(8):
        for f in features:
            new_cols.append(f"{f}_ch{ch}")
    
    df_agg.columns = ["time"] + new_cols
    
    # remove incomplete timestamps
    valid_times = counts[counts == 8].index
    df = df[df["time"].isin(valid_times)]
    
    df.groupby(["time", "channel"]).size().value_counts()
    
    # Feature Engineering
    df["doppler_diff"] = df.groupby("channel")["Carrier_Doppler_hz"].diff()
    df["doppler_diff2"] = df.groupby("channel")["doppler_diff"].diff()
    
    df["range_diff"] = df.groupby("channel")["Pseudorange_m"].diff()
    
    df["phase_diff"] = df.groupby("channel")["Carrier_phase"].diff()
    
    window = 5
    
    df["range_roll_mean"] = df.groupby("channel")["Pseudorange_m"].rolling(window).mean().reset_index(0, drop=True)
    df["range_roll_std"]  = df.groupby("channel")["Pseudorange_m"].rolling(window).std().reset_index(0, drop=True)
    
    df["doppler_roll_std"] = df.groupby("channel")["Carrier_Doppler_hz"].rolling(window).std().reset_index(0, drop=True)
    
    k = 3
    
    df["range_jump"] = (abs(df["range_diff"]) > k * df["range_roll_std"]).astype(int)
    df["doppler_jump"] = (abs(df["doppler_diff"]) > k * df["doppler_roll_std"]).astype(int)
    
    ## Signal processing feature
    
    df["corr_symmetry"] = abs(df["EC"] - df["LC"])
    
    df["ec_pc_diff"] = abs(df["EC"] - df["PC"])
    df["lc_pc_diff"] = abs(df["LC"] - df["PC"])
    
    eps = 1e-6
    
    df["power_ratio"] = df["PIP"] / (df["PQP"] + eps)
    df["power_total"] = df["PIP"] + df["PQP"]
    df["power_log_ratio"] = np.log(df["power_ratio"] + eps)
    
    df["time_gap"] = df["RX_time"] - df["TOW"]
    
    df["ec_ratio"] = df["EC"] / (df["PC"] + eps)
    df["lc_ratio"] = df["LC"] / (df["PC"] + eps)
    
    ## Cross satelite feature
    
    grouped = df.groupby("time")
    
    cn0_stats = grouped["CN0"].agg(["mean", "std"]).rename(columns={
        "mean": "cn0_mean",
        "std": "cn0_std"
    })
    
    range_stats = grouped["Pseudorange_m"].agg(["mean", "std"]).rename(columns={
        "mean": "range_mean",
        "std": "range_std"
    })
    
    doppler_stats = grouped["Carrier_Doppler_hz"].agg(["mean", "std"]).rename(columns={
        "mean": "doppler_mean",
        "std": "doppler_std"
    })
    
    tcd_stats = grouped["TCD"].agg(["mean", "std"]).rename(columns={
        "mean": "tcd_mean",
        "std": "tcd_std"
    })
    
    sym_stats = grouped["corr_symmetry"].agg(["mean", "std"]).rename(columns={
        "mean": "sym_mean",
        "std": "sym_std"
    })
    
    ## Merging full dataset
    
    cross_features = pd.concat([
        cn0_stats,
        range_stats,
        doppler_stats,
        tcd_stats,
        sym_stats
    ], axis=1).reset_index()
    
    df_final = df_agg.merge(cross_features, on="time", how="left")


    
    return df_final
