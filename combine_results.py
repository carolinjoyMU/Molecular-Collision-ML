"""
Pool the four trained models (excitation/quenching x ortho-H2/para-H2).

Usage:
    python combine_results.py results/exc_ortho results/exc_para results/que_ortho results/que_para
"""

import sys
import numpy as np

ALL_U_VALUES = [170.479, 346.41, 703.89, 1430.0, 2906.33, 5906.0, 12000.0]


def main():
    result_dirs = sys.argv[1:]
    if not result_dirs:
        result_dirs = ['results/exc_ortho', 'results/exc_para', 'results/que_ortho', 'results/que_para']

    y_true, y_pred, U = [], [], []
    for d in result_dirs:
        data = np.load(f'{d}/preds.npz')
        y_true.append(data['y_true']); y_pred.append(data['y_pred']); U.append(data['U_vals'])
    y_true, y_pred, U = np.concatenate(y_true), np.concatenate(y_pred), np.concatenate(U)

    yh, yt = np.exp(y_pred), np.exp(y_true)

    print('=' * 40)
    print('H2O + H2 cross section NN — final results')
    print('=' * 40)
    overall = 100*np.sqrt(np.mean(((yh-yt)/yt)**2))
    print(f'Overall RRMSE ({len(yt):,} transitions): {overall:.1f}%\n')
    for u in ALL_U_VALUES:
        m = (U == u)
        if m.sum() == 0:
            continue
        r = 100*np.sqrt(np.mean(((yh[m]-yt[m])/yt[m])**2))
        print(f'  U = {u:>9.1f} cm-1  ({m.sum():>7,} transitions):  RRMSE = {r:.1f}%')


if __name__ == '__main__':
    main()
