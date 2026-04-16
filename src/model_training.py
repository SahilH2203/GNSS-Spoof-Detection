def run_model(df_final):

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    import torch
    import torch.nn as nn

  
    df_final = df_final.fillna(0)
    
    # Building Model
    
    ## Baseline Model
    
    X = df_final.drop(columns=["time"])
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )
    
    iso.fit(X_scaled)
    
    scores = iso.decision_function(X_scaled)   # higher = normal
    anomaly_scores = -scores                   # higher = more anomalous
    
    threshold = np.percentile(anomaly_scores, 98)  
    
    df_final["spoofed"] = (anomaly_scores > threshold).astype(int)
    
    k = 10  
    
    df_final["confidence"] = 1 / (1 + np.exp(-k * (anomaly_scores - threshold)))
    
    ## Rule base
    
    # Rule 1: correlator symmetry
    sym_thresh = df["corr_symmetry"].quantile(0.98)
    
    # Rule 2: CN0 high anomaly
    cn0_thresh = df["CN0"].quantile(0.98)
    
    # Aggregate rule per timestamp
    rule_flags = df.groupby("time").apply(
        lambda g: int(
            (g["corr_symmetry"].mean() > sym_thresh) or
            (g["CN0"].mean() > cn0_thresh)
        )
    ).reset_index(name="rule_spoofed")
    
    df_final = df_final.merge(rule_flags, on="time", how="left")
    
    df_final["final_spoofed"] = (
        (df_final["spoofed"] == 1) | 
        (df_final["rule_spoofed"] == 1)
    ).astype(int)
    
    
    ## Intermediate
    
    X = df_final.drop(columns=["time", "spoofed", "confidence"], errors="ignore").values
    scaler_dl = StandardScaler()
    X_scaled = scaler_dl.fit_transform(X)
    
    def create_sequences(data, window=20):
        sequences = []
        for i in range(len(data) - window):
            sequences.append(data[i:i+window])
        return np.array(sequences)
    
    window_size = 20
    X_seq = create_sequences(X_scaled, window_size)

    
    ### LSTM Encoder

    
    class LSTMAutoencoder(nn.Module):
        def __init__(self, n_features, hidden_size=64):
            super().__init__()
            
            self.encoder = nn.LSTM(n_features, hidden_size, batch_first=True)
            self.decoder = nn.LSTM(hidden_size, n_features, batch_first=True)
    
        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            hidden_repeated = hidden.repeat(x.size(1), 1, 1).permute(1,0,2)
            decoded, _ = self.decoder(hidden_repeated)
            return decoded
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    
    model = LSTMAutoencoder(n_features=X_seq.shape[2]).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, X_tensor)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    model.eval()
    
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon)**2, dim=(1,2)).cpu().numpy()
    
    df_final["lstm_score"] = 0
    df_final.loc[window_size:, "lstm_score"] = errors
    
    threshold_lstm = errors.mean() + 2 * errors.std()
    
    df_final["lstm_spoofed"] = (df_final["lstm_score"] > threshold_lstm).astype(int)
    
    df_final["final_spoofed"] = (
        (df_final["spoofed"] == 1) |
        (df_final["lstm_spoofed"] == 1)
    ).astype(int)
    
    # combine scores
    combined_score = anomaly_scores + df_final["lstm_score"]
    
    # normalize
    combined_score = (combined_score - combined_score.min()) / (combined_score.max() - combined_score.min())
    
    df_final["confidence"] = combined_score
    
    submission1 = df_final[["time", "final_spoofed", "confidence"]].copy()
    
    submission1.columns = ["time", "spoofed", "confidence"]
    
    submission1.to_csv("submission_lstm.csv", index=False)
    
    ### CNN

    class CNNAutoencoder(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Conv1d(n_features, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 32, kernel_size=3, padding=1),
                nn.ReLU()
            )
            
            self.decoder = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, n_features, kernel_size=3, padding=1)
            )
    
        def forward(self, x):
            x = x.permute(0, 2, 1)  # (batch, features, time)
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded.permute(0, 2, 1)
    
    model_cnn = CNNAutoencoder(n_features=X_seq.shape[2]).to(device)
    
    optimizer = torch.optim.Adam(model_cnn.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    for epoch in range(10):
        optimizer.zero_grad()
        output = model_cnn(X_tensor)
        loss = criterion(output, X_tensor)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    with torch.no_grad():
        recon = model_cnn(X_tensor)
        cnn_errors = torch.mean((X_tensor - recon)**2, dim=(1,2)).cpu().numpy()
    
    df_final["cnn_score"] = 0
    df_final.loc[window_size:, "cnn_score"] = cnn_errors
    
    threshold_cnn = cnn_errors.mean() + 2 * cnn_errors.std()
    
    df_final["cnn_spoofed"] = (df_final["cnn_score"] > threshold_cnn).astype(int)
    
    df_final["vote_sum"] = (
        df_final["spoofed"] + 
        df_final["lstm_spoofed"] + 
        df_final["cnn_spoofed"]
    )
    
    df_final["final_spoofed"] = (df_final["vote_sum"] >= 2).astype(int)
    
    df_final["vote_sum"].head()
    
    df_final["confidence"] = df_final["vote_sum"] / 3
    
    submission2 = df_final[["time", "final_spoofed", "confidence"]].copy()
    
    submission2.columns = ["time", "spoofed", "confidence"]
    
    df_final["final_spoofed"].value_counts()
    
    ## Combining
    
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    df_final["iso_score"] = normalize(anomaly_scores)
    df_final["lstm_score"] = normalize(df_final["lstm_score"])
    df_final["cnn_score"] = normalize(df_final["cnn_score"])
    
    df_final["rule_score"] = df_final["rule_spoofed"].astype(float)
    
    w_iso  = 0.3
    w_lstm = 0.4
    w_cnn  = 0.2
    w_rule = 0.1
    
    df_final["fusion_score"] = (
        w_iso  * df_final["iso_score"] +
        w_lstm * df_final["lstm_score"] +
        w_cnn  * df_final["cnn_score"] +
        w_rule * df_final["rule_score"]
    )
    
    threshold = df_final["fusion_score"].mean() + 2 * df_final["fusion_score"].std()
    
    df_final["final_spoofed"] = (df_final["fusion_score"] > threshold).astype(int)
    
    df_final["confidence"] = 1 / (1 + np.exp(-10 * (df_final["fusion_score"] - threshold)))
    
    window = 5
    
    df_final["final_spoofed"] = (
        df_final["final_spoofed"]
        .rolling(window, center=True)
        .max()
        .fillna(0)
        .astype(int)
    )
    
    submission3 = df_final[["time", "final_spoofed", "confidence"]]
    submission3.columns = ["time", "spoofed", "confidence"]
    
    submission3.to_csv("final_submission.csv", index=False)
    
    submission3['spoofed'].value_counts()
    
    
        
    submission = df_final[["time", "final_spoofed", "confidence"]]
    submission.columns = ["time", "spoofed", "confidence"]
        
        return submission
