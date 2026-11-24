"""
ETF 모멘텀 전략 백테스트 — 체결 시점 재검증 + 블록 부트스트랩

학습 구간(2016~2021)에서 36개 조합 중 샤프지수 1위를 고르고, 검증 구간(2022~2025)에 그대로 적용합니다.
두 가지 체결 가정으로 나란히 돌립니다.
  - 당일 체결: 월말 종가로 신호 계산, 같은 월말 종가에 매매
  - 익일 체결: 월말 종가로 신호 계산, 다음 거래일(익월 첫 거래일) 종가에 매매
부트스트랩은 독립 재추출과 함께, 월별 자기상관을 반영한 블록 부트스트랩(6개월 블록)도 계산합니다.

실행:
    pip install -r requirements.txt
    python momentum_backtest.py          # 실제 데이터 (Yahoo Finance)
    python momentum_backtest.py --demo   # 인터넷 없이 가짜 데이터로 동작만 확인
출력: 콘솔 요약, backtest_grid.csv, backtest_result.png
"""
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Glyph")

# ---------------- 설정 ----------------
TICKERS = {
    "069500.KS": "KODEX 200",
    "133690.KS": "TIGER 미국나스닥100",
    "143850.KS": "TIGER 미국S&P500선물(H)",
    "195930.KS": "TIGER 유로스탁스50(H)",
    "148070.KS": "KOSEF 국고채10년",
    "132030.KS": "KODEX 골드선물(H)",
    "138230.KS": "KOSEF 미국달러선물",
}
DATA_START = "2015-01-01"   # 12개월 룩백 계산용 여유 1년
DATA_END = "2026-01-31"     # 2025년 12월 신호의 익일 체결가(2026년 1월 첫 거래일)까지 필요
IS_START, IS_END = "2016-01-01", "2021-12-31"    # 학습(최적화) 구간
OOS_START, OOS_END = "2022-01-01", "2025-12-31"  # 검증 구간
LOOKBACKS = range(1, 13)
TOPKS = [1, 2, 3]
COST = 0.001        # 매매 회전율 1당 0.1%
N_BOOT = 1000
BLOCK = 6
SEED = 42


# ---------------- 데이터 ----------------
def load_daily(demo=False):
    if demo:
        rng = np.random.default_rng(0)
        idx = pd.date_range(DATA_START, DATA_END, freq="B")
        rets = rng.normal(0.0003, 0.01, size=(len(idx), len(TICKERS)))
        return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=idx,
                            columns=list(TICKERS.values()))
    import yfinance as yf
    raw = yf.download(list(TICKERS), start=DATA_START, end=DATA_END,
                      auto_adjust=True, progress=False)["Close"]
    return raw.rename(columns=TICKERS)


def build_monthly(daily):
    """월말 신호가격(signal)과 익월 첫 거래일 체결가격(nextday)을 같은 월 인덱스로 정렬"""
    daily = daily.dropna(how="all")
    g = daily.groupby(daily.index.to_period("M"))
    signal = g.last()
    nextday = g.first().shift(-1)          # t월 신호 → (t+1)월 첫 거래일 종가에 체결
    for df in (signal, nextday):
        df.index = df.index.to_timestamp(how="end").normalize()

    empty = signal.columns[signal.isna().all()]
    if len(empty):
        print(f"[알림] 다운로드 실패로 제외: {list(empty)}")
    first_ok = pd.to_datetime(signal.drop(columns=empty).apply(lambda s: s.first_valid_index()))
    late = first_ok[first_ok > pd.Timestamp(DATA_START) + pd.offsets.MonthEnd(2)].index
    if len(late):
        print(f"[알림] 시작 시점 데이터 부족으로 제외: {list(late)}")
    drop = list(empty) + list(late)
    signal, nextday = signal.drop(columns=drop), nextday.drop(columns=drop)
    if signal.shape[1] < 3:
        sys.exit("데이터를 받은 종목이 3개 미만입니다. 인터넷 연결이나 yfinance 버전을 확인하세요.")
    signal = signal.dropna()
    nextday = nextday.reindex(signal.index)
    print(f"사용 종목 {signal.shape[1]}개, 신호 기간 {signal.index[0]:%Y-%m} ~ {signal.index[-1]:%Y-%m}")
    return signal, nextday


# ---------------- 백테스트 ----------------
def strategy_returns(signal, execp, lookback, topk):
    """t월 말 signal로 비중 결정 → execp[t]에서 매수, execp[t+1]까지 보유. 수익은 t+1월에 기록"""
    mom = signal.pct_change(lookback)
    ranks = mom.rank(axis=1, ascending=False, method="first")
    w = (ranks <= topk).astype(float) / topk
    w[mom.isna().any(axis=1)] = np.nan
    rets = execp.pct_change()
    turnover = (w - w.shift()).abs().sum(axis=1, min_count=1)
    port = (w.shift() * rets).sum(axis=1, min_count=1) - COST * turnover.shift().fillna(0)
    port[w.shift().isna().any(axis=1) | rets.isna().any(axis=1)] = np.nan
    return port


def benchmark_returns(execp):
    return execp.pct_change().mean(axis=1)   # 전 종목 동일비중, 매월 리밸런싱


def metrics(r):
    r = r.dropna()
    cagr = (1 + r).prod() ** (12 / len(r)) - 1
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan  # 무위험수익률 0 가정
    curve = (1 + r).cumprod()
    mdd = (curve / curve.cummax() - 1).min()
    return cagr, sharpe, mdd


def boot_win(pair, block=1, n=N_BOOT, seed=SEED):
    """pair: (T, 2) 같은 달 [전략, 벤치마크]. block=1이면 독립 재추출, >1이면 이동 블록 부트스트랩"""
    rng = np.random.default_rng(seed)
    T = len(pair)
    nb = int(np.ceil(T / block))
    starts = rng.integers(0, T - block + 1, size=(n, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :T]
    s = pair[idx]
    return (s[:, :, 0].mean(axis=1) > s[:, :, 1].mean(axis=1)).mean()


def run(signal, execp, label):
    bench = benchmark_returns(execp)
    rows, series = [], {}
    for L in LOOKBACKS:
        for K in TOPKS:
            if K >= signal.shape[1]:
                continue
            r = strategy_returns(signal, execp, L, K)
            series[(L, K)] = r
            is_m, oos_m = metrics(r[IS_START:IS_END]), metrics(r[OOS_START:OOS_END])
            rows.append(dict(exec=label, lookback=L, topk=K,
                             IS_CAGR=is_m[0], IS_Sharpe=is_m[1], IS_MDD=is_m[2],
                             OOS_CAGR=oos_m[0], OOS_Sharpe=oos_m[1], OOS_MDD=oos_m[2]))
    grid = pd.DataFrame(rows)
    grid["OOS_rank"] = grid["OOS_Sharpe"].rank(ascending=False).astype(int)
    best = grid.loc[grid["IS_Sharpe"].idxmax()]
    key = (int(best.lookback), int(best.topk))
    pair = pd.concat([series[key], bench], axis=1)[OOS_START:OOS_END].dropna().values
    return dict(label=label, grid=grid, series=series, bench=bench, best=best, key=key,
                b_is=metrics(bench[IS_START:IS_END]), b_oos=metrics(bench[OOS_START:OOS_END]),
                rho=grid["IS_Sharpe"].corr(grid["OOS_Sharpe"], method="spearman"),
                win_iid=boot_win(pair, 1), win_blk=boot_win(pair, BLOCK), n_oos=len(pair))


def report(res):
    b, (L, K) = res["best"], res["key"]
    print("=" * 64)
    print(f"[{res['label']}] 학습 샤프 1위 조합: 룩백 {L}개월, 상위 {K}개")
    print(f"{'':12}{'학습(16~21)':>16}{'검증(22~25)':>16}")
    print(f"{'CAGR':12}{b.IS_CAGR:>16.1%}{b.OOS_CAGR:>16.1%}")
    print(f"{'샤프지수':10}{b.IS_Sharpe:>16.2f}{b.OOS_Sharpe:>16.2f}")
    print(f"{'MDD':12}{b.IS_MDD:>16.1%}{b.OOS_MDD:>16.1%}")
    print(f"벤치마크(동일비중) 샤프: 학습 {res['b_is'][1]:.2f} / 검증 {res['b_oos'][1]:.2f}")
    print(f"검증 구간 순위: {len(res['grid'])}개 중 {int(b.OOS_rank)}위 | "
          f"검증 샤프 중앙값 {res['grid'].OOS_Sharpe.median():.2f} | 순위상관 {res['rho']:.2f}")
    print(f"부트스트랩(검증 {res['n_oos']}개월) 월평균 수익이 벤치마크보다 높았던 비율: "
          f"독립 {res['win_iid']:.0%} / {BLOCK}개월 블록 {res['win_blk']:.0%}")


def main():
    signal, nextday = build_monthly(load_daily("--demo" in sys.argv))
    same = run(signal, signal, "당일 체결")
    nxt = run(signal, nextday, "익일 체결")
    print()
    report(same)
    report(nxt)

    # 당일 체결 1위 조합을 익일 체결로 그대로 돌렸을 때
    k = same["key"]
    g = nxt["grid"].set_index(["lookback", "topk"]).loc[k]
    print("=" * 64)
    print(f"당일 체결 1위 조합(L={k[0]}, K={k[1]})을 익일 체결로 적용: "
          f"학습 샤프 {g.IS_Sharpe:.2f} → 검증 샤프 {g.OOS_Sharpe:.2f} (검증 {int(g.OOS_rank)}위)")
    print("=" * 64)

    pd.concat([same["grid"], nxt["grid"]]).to_csv("backtest_grid.csv", index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = ["NanumGothic", "Malgun Gothic", "AppleGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for res, c in ((same, "tab:blue"), (nxt, "tab:orange")):
        gr = res["grid"]
        ax[0].scatter(gr.IS_Sharpe, gr.OOS_Sharpe, alpha=0.5, color=c, label=f"{res['label']} 36개 조합")
        ax[0].scatter(res["best"].IS_Sharpe, res["best"].OOS_Sharpe, color=c, s=120,
                      edgecolor="black", label=f"{res['label']} 학습 1위")
    ax[0].axhline(nxt["b_oos"][1], ls="--", color="gray", label="벤치마크 검증 샤프(익일)")
    ax[0].set_xlabel("학습 구간 샤프지수"); ax[0].set_ylabel("검증 구간 샤프지수")
    ax[0].set_title("체결 시점별 학습 vs 검증 성과"); ax[0].legend(fontsize=8)

    for res, c in ((same, "tab:blue"), (nxt, "tab:orange")):
        cum = (1 + res["series"][res["key"]][IS_START:OOS_END].fillna(0)).cumprod()
        ax[1].plot(cum, color=c, label=f"{res['label']} 학습 1위 (L={res['key'][0]}, K={res['key'][1]})")
    ax[1].plot((1 + nxt["bench"][IS_START:OOS_END].fillna(0)).cumprod(), color="gray", label="동일비중(익일)")
    ax[1].axvline(pd.Timestamp(OOS_START), ls=":", color="black")
    ax[1].set_title("누적 수익 (점선 이후 검증 구간)"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig("backtest_result.png", dpi=150)
    print("저장: backtest_grid.csv, backtest_result.png")


if __name__ == "__main__":
    main()
