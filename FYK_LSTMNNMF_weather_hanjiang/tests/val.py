# =========================
# ✅ 只需要改这里的路径/参数
# =========================
WQ_PATH   = "/home/fanyunkai/FYK_data/WQ_hanjiang/WQ_hanjiang.npy"
REC_PATH  = "/home/fanyunkai/FYK_data/WQ_hanjiang_weatherresults/reconstructed_matrix.npy"
TRAIN_PATH= "/home/fanyunkai/FYK_data/WQ_hanjiang/train_mask.npy"
VAL_PATH  = "/home/fanyunkai/FYK_data/WQ_hanjiang/val_mask.npy"

OUT_DIR   = "/home/fanyunkai/FYK_data/WQ_hanjiang_weatherresults/plots"   # 输出图片文件夹
STATION_NAMES = None  # 例如 ["S1","S2","S3","S4","S5","S6"]；不填则用 site_0...site_n
INVERT_MASK = False   # mask 语义若相反（True表示缺失/不可用）就设 True

# ✅ 关键：让 gt/mask 自动对齐 pred 的时间长度（pred=训练段）
USE_TRAIN_SPLIT = True

# Plotly 时序图参数（交互式 html）
MAKE_PLOTLY_TS = True
VAL_STYLE = "rug"     # "rug"（推荐，快）或 "band"（更显眼但可能慢）
DOWNSAMPLE = 1        # 若打开卡，把它改成 5 或 10
SHOW_ONLY_KNOWN = True
# =========================


import os
import numpy as np
import matplotlib.pyplot as plt

# Plotly 是可选的：MAKE_PLOTLY_TS=True 才需要
if MAKE_PLOTLY_TS:
    import plotly.graph_objects as go


def _as_bool_mask(mask, invert=False):
    """Accepts bool mask or 0/1 mask; returns bool."""
    m = mask if mask.dtype == np.bool_ else mask.astype(bool)
    return ~m if invert else m


def _align_to_pred_len(gt, pred, train_mask, val_mask):
    """
    pred: [sites, T_pred]
    将 gt/train/val 统一切到 pred 的时间长度（默认认为 pred 对应前段训练期）
    """
    T_pred = pred.shape[1]
    T_gt = gt.shape[1]
    if T_gt == T_pred:
        return gt, train_mask, val_mask
    if T_gt < T_pred:
        raise ValueError(f"gt shorter than pred: gt T={T_gt}, pred T={T_pred}")

    gt2 = gt[:, :T_pred]
    tr2 = train_mask[:, :T_pred]
    va2 = val_mask[:, :T_pred]
    print(f"[Align] pred T={T_pred}, gt/masks T={T_gt} -> sliced gt/train/val to [:,{T_pred}]")
    return gt2, tr2, va2


def _safe_xy(gt, pred, mask):
    """Return 1D x,y filtered by mask and finite values (ignores natural missing NaNs)."""
    m = mask & np.isfinite(gt) & np.isfinite(pred)
    x = gt[m].astype(np.float64).ravel()
    y = pred[m].astype(np.float64).ravel()
    return x, y


def _metrics(x, y):
    if x.size == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan}
    err = y - x
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    ss_res = np.sum(err**2)
    ss_tot = np.sum((x - np.mean(x))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": int(x.size), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}


def _scatter_xy(x, y, title, out_path):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=8, alpha=0.4)
    if x.size > 0 and y.size > 0:
        lo = np.nanmin([np.min(x), np.min(y)])
        hi = np.nanmax([np.max(x), np.max(y)])
        plt.plot([lo, hi], [lo, hi], linewidth=1)  # y=x
        plt.xlim(lo, hi)
        plt.ylim(lo, hi)
    plt.xlabel("Observed (x)")
    plt.ylabel("Predicted (y)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def _timeseries_site_matplotlib(gt_row, pred_row, train_mask_row, val_mask_row, title, out_path):
    """
    静态版（matplotlib）：
    - 黑点：实测
    - 红点：预测
    - val 位置：竖虚线（如果 val 点多会很糊，所以建议用 plotly 或分段）
    """
    T = gt_row.shape[0]
    t = np.arange(T)

    known_mask = (train_mask_row | val_mask_row) & np.isfinite(gt_row) & np.isfinite(pred_row)

    plt.figure(figsize=(12, 4))
    plt.scatter(t[known_mask], gt_row[known_mask], s=10, alpha=0.85, c="k", label="Observed")
    plt.scatter(t[known_mask], pred_row[known_mask], s=10, alpha=0.55, c="r", label="Predicted")

    for tt in np.where(val_mask_row)[0]:
        plt.axvline(tt, linestyle="--", linewidth=0.6, alpha=0.12, color="k")

    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close()


def plot_timeseries_plotly(
    gt_row, pred_row, train_mask_row, val_mask_row,
    title, out_path_html,
    show_only_known=True,
    val_style="rug",    # "rug"（推荐）或 "band"
    downsample=1
):
    """
    交互式 Plotly：
    - 黑点：实测
    - 红点：预测
    - val 位置：rug（顶部刻度）/ band（半透明竖条）
    """
    gt_row = np.asarray(gt_row)
    pred_row = np.asarray(pred_row)
    train_mask_row = np.asarray(train_mask_row).astype(bool)
    val_mask_row = np.asarray(val_mask_row).astype(bool)

    T = gt_row.shape[0]
    t = np.arange(T)

    if show_only_known:
        known = (train_mask_row | val_mask_row) & np.isfinite(gt_row) & np.isfinite(pred_row)
    else:
        known = np.isfinite(gt_row) & np.isfinite(pred_row)

    t_k = t[known]
    gt_k = gt_row[known]
    pred_k = pred_row[known]

    if downsample > 1 and t_k.size > 0:
        idx = np.arange(0, t_k.size, downsample)
        t_k = t_k[idx]
        gt_k = gt_k[idx]
        pred_k = pred_k[idx]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=t_k, y=gt_k,
        mode="markers",
        name="Observed",
        marker=dict(size=5, color="black"),
        hovertemplate="t=%{x}<br>obs=%{y}<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=t_k, y=pred_k,
        mode="markers",
        name="Predicted",
        marker=dict(size=5, color="red", opacity=0.6),
        hovertemplate="t=%{x}<br>pred=%{y}<extra></extra>"
    ))

    val_times = np.where(val_mask_row)[0]

    shapes = []
    if val_style == "band":
        # 半透明竖条（val 多的话会慢）
        for tt in val_times:
            shapes.append(dict(
                type="rect",
                xref="x", yref="paper",
                x0=tt - 0.5, x1=tt + 0.5,
                y0=0, y1=1,
                fillcolor="rgba(0,0,0,0.06)",
                line=dict(width=0)
            ))
    else:
        # 顶部 rug（推荐）
        for tt in val_times:
            shapes.append(dict(
                type="line",
                xref="x", yref="paper",
                x0=tt, x1=tt,
                y0=1.0, y1=0.96,
                line=dict(color="rgba(0,0,0,0.35)", width=1)
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Time index",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes
    )

    os.makedirs(os.path.dirname(out_path_html), exist_ok=True)
    fig.write_html(out_path_html)


def validate_and_plot(WQ_hanjiang, reconstructed_matrix, train_mask, val_mask,
                      out_dir=OUT_DIR, station_names=STATION_NAMES):
    os.makedirs(out_dir, exist_ok=True)

    gt = np.asarray(WQ_hanjiang)
    pred = np.asarray(reconstructed_matrix)

    train_mask = _as_bool_mask(np.asarray(train_mask), invert=INVERT_MASK)
    val_mask   = _as_bool_mask(np.asarray(val_mask),   invert=INVERT_MASK)

    # 对齐到 pred 的时间长度（训练段）
    if USE_TRAIN_SPLIT:
        gt, train_mask, val_mask = _align_to_pred_len(gt, pred, train_mask, val_mask)

    assert gt.shape == pred.shape == train_mask.shape == val_mask.shape, \
        f"Shape mismatch: gt{gt.shape}, pred{pred.shape}, train{train_mask.shape}, val{val_mask.shape}"

    num_sites, _ = gt.shape
    if station_names is None:
        station_names = [f"site_{i}" for i in range(num_sites)]
    else:
        assert len(station_names) == num_sites, "station_names length must equal num_sites"

    # ========== 1) 全局散点：train / val ==========
    x_tr, y_tr = _safe_xy(gt, pred, train_mask)
    m_tr = _metrics(x_tr, y_tr)
    _scatter_xy(
        x_tr, y_tr,
        title=f"GLOBAL TRAIN scatter (n={m_tr['n']}, RMSE={m_tr['rmse']:.4f}, MAE={m_tr['mae']:.4f}, R2={m_tr['r2']:.4f})",
        out_path=os.path.join(out_dir, "global_train_scatter.png")
    )

    x_va, y_va = _safe_xy(gt, pred, val_mask)
    m_va = _metrics(x_va, y_va)
    _scatter_xy(
        x_va, y_va,
        title=f"GLOBAL VAL scatter (n={m_va['n']}, RMSE={m_va['rmse']:.4f}, MAE={m_va['mae']:.4f}, R2={m_va['r2']:.4f})",
        out_path=os.path.join(out_dir, "global_val_scatter.png")
    )

    # ========== 2) 分站点散点：train / val ==========
    per_site_dir = os.path.join(out_dir, "per_site_scatter")
    os.makedirs(per_site_dir, exist_ok=True)

    for i in range(num_sites):
        name = station_names[i]

        x_tr_i, y_tr_i = _safe_xy(gt[i], pred[i], train_mask[i])
        mi_tr = _metrics(x_tr_i, y_tr_i)
        _scatter_xy(
            x_tr_i, y_tr_i,
            title=f"{name} TRAIN scatter (n={mi_tr['n']}, RMSE={mi_tr['rmse']:.4f}, MAE={mi_tr['mae']:.4f}, R2={mi_tr['r2']:.4f})",
            out_path=os.path.join(per_site_dir, f"{name}_train_scatter.png")
        )

        x_va_i, y_va_i = _safe_xy(gt[i], pred[i], val_mask[i])
        mi_va = _metrics(x_va_i, y_va_i)
        _scatter_xy(
            x_va_i, y_va_i,
            title=f"{name} VAL scatter (n={mi_va['n']}, RMSE={mi_va['rmse']:.4f}, MAE={mi_va['mae']:.4f}, R2={mi_va['r2']:.4f})",
            out_path=os.path.join(per_site_dir, f"{name}_val_scatter.png")
        )

    # ========== 3) 分站点时间序列 ==========
    if MAKE_PLOTLY_TS:
        ts_dir = os.path.join(out_dir, "per_site_timeseries_plotly")
        os.makedirs(ts_dir, exist_ok=True)

        for i in range(num_sites):
            name = station_names[i]
            plot_timeseries_plotly(
                gt_row=gt[i],
                pred_row=pred[i],
                train_mask_row=train_mask[i],
                val_mask_row=val_mask[i],
                title=f"{name} interactive time series (black=obs, red=pred; VAL={VAL_STYLE})",
                out_path_html=os.path.join(ts_dir, f"{name}_timeseries.html"),
                show_only_known=SHOW_ONLY_KNOWN,
                val_style=VAL_STYLE,
                downsample=DOWNSAMPLE
            )
    else:
        ts_dir = os.path.join(out_dir, "per_site_timeseries_matplotlib")
        os.makedirs(ts_dir, exist_ok=True)

        for i in range(num_sites):
            name = station_names[i]
            _timeseries_site_matplotlib(
                gt_row=gt[i],
                pred_row=pred[i],
                train_mask_row=train_mask[i],
                val_mask_row=val_mask[i],
                title=f"{name} time series (black=obs, red=pred; dashed=VAL positions)",
                out_path=os.path.join(ts_dir, f"{name}_timeseries.png")
            )

    print("Saved plots to:", os.path.abspath(out_dir))
    print("GLOBAL TRAIN:", m_tr)
    print("GLOBAL VAL  :", m_va)


def main():
    WQ_hanjiang = np.load(WQ_PATH)
    reconstructed_matrix = np.load(REC_PATH)
    train_mask = np.load(TRAIN_PATH)
    val_mask = np.load(VAL_PATH)

    validate_and_plot(
        WQ_hanjiang=WQ_hanjiang,
        reconstructed_matrix=reconstructed_matrix,
        train_mask=train_mask,
        val_mask=val_mask,
        out_dir=OUT_DIR,
        station_names=STATION_NAMES
    )


if __name__ == "__main__":
    main()
