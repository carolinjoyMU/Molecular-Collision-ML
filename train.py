"""
Neural network prediction of H2O + H2 rotational state-to-state transition
cross sections (Joy, Occhiogrosso, Mandal, Babikov).

Plain MLP, 13 input features -> 4 hidden layers of 128 nodes (GELU) -> 1 output
(log cross section). Trained with AdamW + Huber loss in log-space.

The four subsets (excitation/quenching x ortho-H2/para-H2) are
trained as separate models. Run all four, then use combine_results.py to pool them into the overall + per-U RRMSE table.

Usage:
    python train.py --process excitation --h2_symmetry ortho --out_dir results/exc_ortho
    python train.py --process excitation --h2_symmetry para  --out_dir results/exc_para
    python train.py --process quenching  --h2_symmetry ortho --out_dir results/que_ortho
    python train.py --process quenching  --h2_symmetry para  --out_dir results/que_para
"""

import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ALL_U_VALUES = [170.479, 346.41, 703.89, 1430.0, 2906.33, 5906.0, 12000.0]
FEAT_COLS    = ['WEI','WEF','WJI','WJF','WKaI','WKaF','WKcI','WKcF',
                'HEI','HEF','HJI','HJF','U_file']   # 13 raw features


_SWAP = {'WEI':'WEF','WEF':'WEI','WJI':'WJF','WJF':'WJI',
         'WKaI':'WKaF','WKaF':'WKaI','WKcI':'WKcF','WKcF':'WKcI',
         'HEI':'HEF','HEF':'HEI','HJI':'HJF','HJF':'HJI'}


def build_rev_features(df):
    cols = [df[_SWAP.get(c, c)].values for c in FEAT_COLS]
    return np.column_stack(cols).astype(np.float32)


def build_model(d=128, n_blocks=3):
    """13 -> (n_blocks+1) hidden layers of width d -> 1, plain MLP 4 hidden layers of 128."""
    layers = [nn.Linear(13, d), nn.GELU()]
    for _ in range(n_blocks):
        layers += [nn.Linear(d, d), nn.GELU()]
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers).to(device)


def load_data(data_folder, process, h2_symmetry, cutoff_log, min_u):
    fname = 'excitation_transitions.csv' if process == 'excitation' else 'quenching_transitions.csv'
    df = pd.read_csv(os.path.join(data_folder, fname))
    df = df.rename(columns={
        'E_initial_H2O':'WEI', 'E_final_H2O':'WEF',
        'H2O_i_j':'WJI', 'H2O_f_j':'WJF',
        'H2O_i_ka':'WKaI', 'H2O_f_ka':'WKaF',
        'H2O_i_kc':'WKcI', 'H2O_f_kc':'WKcF',
        'E_initial_H2':'HEI', 'E_final_H2':'HEF',
        'H2_i_j':'HJI', 'H2_f_j':'HJF',
        'cross_section':'CS',
    })
    df = df[df['CS'] > 0].copy()
    df['CS_log'] = np.log(df['CS'])
    df = df[df['CS_log'] >= cutoff_log]
    df = df[df['U_file'] >= min_u]
    if h2_symmetry == 'ortho':
        df = df[(df['HJI'] % 2 == 1) & (df['HJF'] % 2 == 1)]
    else:
        df = df[(df['HJI'] % 2 == 0) & (df['HJF'] % 2 == 0)]

    df['s_group'] = df['WEI'] + df['HEI']
    for col in ['WEI','WEF','HEI','HEF']:
        df[col] = np.log10(df[col] + 1.0)
    return df.reset_index(drop=True)


def sample_by_initial_state(df, n_samples):
    """Sort transitions by initial-state rotational energy, sample n_samples
    initial states per U (always including the lowest and highest), take all
    final-state transitions for each sampled initial state."""
    train_parts, test_parts = [], []
    for u_val in ALL_U_VALUES:
        u_df = df[df['U_file'] == u_val]
        if len(u_df) == 0:
            continue
        groups = u_df.groupby('s_group')
        keys = sorted(groups.groups.keys())
        s = min(n_samples, len(keys) - 1)
        train_idx = set(np.linspace(0, len(keys) - 1, s, dtype=int))
        tr_keys = [keys[i] for i in sorted(train_idx)]
        te_keys = [keys[i] for i in range(len(keys)) if i not in train_idx]
        train_parts.append(pd.concat([groups.get_group(k) for k in tr_keys]))
        test_parts.append(pd.concat([groups.get_group(k) for k in te_keys]))
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def train_and_eval(tr_df, te_df, d, n_blocks, epochs, patience, lr, seed=42):
    torch.manual_seed(seed)
    X_tr = tr_df[FEAT_COLS].values.astype(np.float32)
    y_tr = tr_df['CS_log'].values.astype(np.float32)
    X_te = te_df[FEAT_COLS].values.astype(np.float32)
    y_te = te_df['CS_log'].values.astype(np.float32)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    X_tr_rev = scaler.transform(build_rev_features(tr_df))
    g_row_tr = ((2*tr_df['WJI']+1)*(2*tr_df['HJI']+1)).values.astype(np.float32)
    g_rev_tr = ((2*tr_df['WJF']+1)*(2*tr_df['HJF']+1)).values.astype(np.float32)
    X_te_rev = scaler.transform(build_rev_features(te_df))
    g_row_te = ((2*te_df['WJI']+1)*(2*te_df['HJI']+1)).values.astype(np.float32)
    g_rev_te = ((2*te_df['WJF']+1)*(2*te_df['HJF']+1)).values.astype(np.float32)

    val_n = max(1, int(len(X_tr) * 0.1))
    tr_n  = len(X_tr) - val_n
    full_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(X_tr_rev),
                            torch.tensor(g_row_tr), torch.tensor(g_rev_tr),
                            torch.tensor(y_tr))
    tr_ds, val_ds = random_split(full_ds, [tr_n, val_n],
                                  generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(tr_ds, batch_size=256, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=val_n, shuffle=False)

    model     = build_model(d, n_blocks)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=30)
    criterion = nn.HuberLoss(delta=1.0)

    best_rrmse, best_state, no_imp = float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, xb_rev, g_row_b, g_rev_b, yb in train_loader:
            xb, xb_rev = xb.to(device), xb_rev.to(device)
            g_row_b, g_rev_b, yb = g_row_b.to(device), g_rev_b.to(device), yb.to(device)
            optimizer.zero_grad()
            pred     = model(xb).squeeze()
            pred_rev = model(xb_rev).squeeze()
            sigma_bar = 0.5 * (g_row_b * torch.exp(pred) + g_rev_b * torch.exp(pred_rev))
            criterion(torch.log(sigma_bar / g_row_b), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for xv, xv_rev, g_row_v, g_rev_v, yv in val_loader:
                xv, xv_rev = xv.to(device), xv_rev.to(device)
                g_row_v, g_rev_v, yv = g_row_v.to(device), g_rev_v.to(device), yv.to(device)
                pv, pv_rev = model(xv).squeeze(), model(xv_rev).squeeze()
                sb_v = 0.5 * (g_row_v * torch.exp(pv) + g_rev_v * torch.exp(pv_rev))
                v_rrmse = torch.sqrt(torch.mean(
                    ((sb_v / g_row_v - torch.exp(yv)) / torch.exp(yv)) ** 2)).item()
        scheduler.step(v_rrmse)
        if v_rrmse < best_rrmse:
            best_rrmse, best_state, no_imp = v_rrmse, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
        if epoch % 50 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  val_RRMSE={v_rrmse*100:5.1f}%", flush=True)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_te_t, X_te_rev_t = torch.tensor(X_te).to(device), torch.tensor(X_te_rev).to(device)
        g_row_te_t = torch.tensor(g_row_te).to(device)
        g_rev_te_t = torch.tensor(g_rev_te).to(device)
        p_fwd, p_rev = model(X_te_t).squeeze(), model(X_te_rev_t).squeeze()
        sb_te = 0.5 * (g_row_te_t * torch.exp(p_fwd) + g_rev_te_t * torch.exp(p_rev))
        y_hat = torch.log(sb_te / g_row_te_t).cpu().numpy()

    return y_hat, y_te, te_df['U_file'].values


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_folder', default='data')
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--process',     choices=['excitation','quenching'], required=True)
    p.add_argument('--h2_symmetry', choices=['ortho','para'], required=True)
    p.add_argument('--d',        type=int,   default=128, help='hidden layer width (paper: 128)')
    p.add_argument('--n_blocks', type=int,   default=3,   help='hidden layers = n_blocks+1 (paper: 4)')
    p.add_argument('--epochs',   type=int,   default=2000)
    p.add_argument('--patience', type=int,   default=150)
    p.add_argument('--lr',       type=float, default=5e-4)
    p.add_argument('--cutoff',   type=float, default=-3.0)
    p.add_argument('--min_u',    type=float, default=170.0)
    p.add_argument('--samples',  type=int,   default=165)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_data(args.data_folder, args.process, args.h2_symmetry, args.cutoff, args.min_u)
    opp_process = 'quenching' if args.process == 'excitation' else 'excitation'
    df_opp = load_data(args.data_folder, opp_process, args.h2_symmetry, args.cutoff, args.min_u)

    tr_df, te_df = sample_by_initial_state(df, args.samples)
    tr_df = pd.concat([tr_df, df_opp], ignore_index=True)

    y_hat, y_te, u_arr = train_and_eval(tr_df, te_df, args.d, args.n_blocks,
                                        args.epochs, args.patience, args.lr)

    np.savez_compressed(os.path.join(args.out_dir, 'preds.npz'),
                        y_true=y_te, y_pred=y_hat, U_vals=u_arr)

    yh, yt = np.exp(y_hat), np.exp(y_te)
    overall = 100*np.sqrt(np.mean(((yh-yt)/yt)**2))
    print(f"\n[{args.process}/{args.h2_symmetry}] overall RRMSE = {overall:.1f}%")
    for u in ALL_U_VALUES:
        m = (u_arr == u)
        if m.sum() == 0:
            continue
        r = 100*np.sqrt(np.mean(((yh[m]-yt[m])/yt[m])**2))
        print(f"  U={u:>9.1f} cm-1 : RRMSE = {r:.1f}%")


if __name__ == '__main__':
    main()
