#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Согласованность классических методов DGA-диагностики и правило отказа от заключения.

Методы:  A) треугольник Duval 1 (CH4, C2H4, C2H2);  B) таблица отношений IEC 60599
         (C2H2/C2H4, CH4/H2, C2H4/C2H6).
Правило: заключение (укрупнённый класс PD / D / T) выдаётся, только если A и B совпали по укрупнённому классу;
         иначе "ПРОВЕРКА"; при недостаточных данных (фильтр L) — "НЕТ ДАННЫХ". Подкласс Duval — только справочно.
ВАЖНО: метки набора НЕ используются для обучения. Согласие методов не доказывает
правильность диагноза. Границы зон перед подачей сверить с текстом IEC 60599.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

GAS = {"Hydrogen (H2)": "H2", "Methane (CH4)": "CH4", "Ethylene (C2H4)": "C2H4",
       "Ethane (C2H6)": "C2H6", "Acetylene (C2H2)": "C2H2"}
GASES = ["H2", "CH4", "C2H4", "C2H6", "C2H2"]
SEED, NOISE_BOUNDS, NOISE_REPEATS = 20261028, [0.05, 0.10], 200
COARSE = {"PD": "PD", "D1": "D", "D2": "D", "D1/D2": "D", "T1": "T", "T2": "T", "T3": "T", "DT": "DT"}

# ---------- метод A: Duval Triangle 1 (границы по рис. 1 в Duval, IEEE EIM 2002) ----------
def duval1(ch4, c2h4, c2h2):
    s = ch4 + c2h4 + c2h2
    if not s > 0: return "UND"
    pm, pe, pa = 100*ch4/s, 100*c2h4/s, 100*c2h2/s
    if pm >= 98: return "PD"
    if pa < 4 and pe < 20: return "T1"
    if pa < 4 and pe < 50: return "T2"
    if pa < 15 and pe >= 50: return "T3"
    if pa >= 13 and pe < 23: return "D1"
    if pa >= 13 and (pe < 40 or pa >= 29): return "D2"
    return "DT"

# ---------- метод B: таблица отношений IEC 60599 ----------
def _r(a, b): return a / b if b > 0 else np.nan
def iec_ratio(h2, ch4, c2h4, c2h6, c2h2):
    r2, r1, r5 = _r(c2h2, c2h4), _r(ch4, h2), _r(c2h4, c2h6)
    def lt(x, v): return x < v if x == x else False
    def gt(x, v): return x > v if x == x else False
    def rng(x, a, b): return (a <= x <= b) if x == x else False
    hit = []
    if lt(r1, 0.1) and lt(r5, 0.2): hit.append("PD")
    if gt(r2, 1) and rng(r1, 0.1, 0.5) and gt(r5, 1): hit.append("D1")
    if rng(r2, 0.6, 2.5) and rng(r1, 0.1, 1.0) and gt(r5, 2): hit.append("D2")
    if gt(r1, 1) and lt(r5, 1): hit.append("T1")
    if lt(r2, 0.1) and gt(r1, 1) and rng(r5, 1, 4): hit.append("T2")
    if lt(r2, 0.2) and gt(r1, 1) and gt(r5, 4): hit.append("T3")
    if len(hit) == 1: return hit[0]
    if not hit: return "UND"
    # IEC 60599:2022 допускает перекрытие D1 и D2: двойное отнесение сохраняется, на укрупнённом уровне это класс D.
    if set(hit).issubset({"D1", "D2"}): return "D1/D2"
    return "AMB"   # совпадения между разными укрупнёнными классами остаются неоднозначными

def classify(df):
    a = [duval1(*r) for r in df[["CH4", "C2H4", "C2H2"]].to_numpy()]
    b = [iec_ratio(*r) for r in df[["H2", "CH4", "C2H4", "C2H6", "C2H2"]].to_numpy()]
    return np.array(a, dtype=object), np.array(b, dtype=object)

def agree(a, b, lenient=False):
    ok = np.zeros(len(a), bool)
    for i, (x, y) in enumerate(zip(a, b)):
        if x in ("UND",) or y in ("UND", "AMB"): continue
        cx, cy = COARSE[x], COARSE[y]
        ok[i] = (cx == cy) or (lenient and cx == "DT" and cy in ("D", "T"))
    return ok

def sufficient(df, level):
    """Критерий достаточности газовых данных (реализация проекта, НЕ критерий IEC):
    хотя бы один из пяти газов >= level (единицы исходного файла). Динамики во времени и типа оборудования в наборе нет."""
    return (df[GASES].max(axis=1) >= level).to_numpy()

def three_tier(df, level):
    """1) недостаточно данных -> НЕТ ДАННЫХ; 2) методы применимы -> согласие; 3) итог или ПРОВЕРКА."""
    suff = sufficient(df, level)
    return np.where(~suff, "НЕТ ДАННЫХ", df["agreed_class"]), suff

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (c-h, c+h)

def run_boundary_tests():
    """Регрессионные тесты реализации; не заменяют сверку зон с первичным рисунком Duval и текстом IEC."""
    assert duval1(98, 1, 1) == "PD"
    assert duval1(97, 2, 1) == "T1"      # %C2H2 < 4, %C2H4 < 20
    assert duval1(75, 22, 3) == "T2"     # %C2H2 < 4, 20 <= %C2H4 < 50
    assert duval1(45, 52, 3) == "T3"     # %C2H2 < 15, %C2H4 >= 50
    assert duval1(70, 10, 20) == "D1"    # %C2H2 >= 13, %C2H4 < 23
    assert duval1(40, 30, 30) == "D2"    # %C2H2 >= 13, 23 <= %C2H4 < 40
    assert duval1(30, 40, 30) == "D2"    # %C2H4 >= 40 и %C2H2 >= 29
    assert duval1(70, 25, 5) == "DT"     # %C2H2 между 4 и 13 при %C2H4 >= 20 -> смешанная зона
    assert duval1(60, 10, 30) == "D1"    # %C2H4 < 23 -> D1 (не D2)
    assert duval1(0, 0, 0) == "UND"
    assert iec_ratio(10, 2, 10, 2, 15) == "D1/D2"    # перекрытие D1 и D2 -> двойное отнесение, укрупнённо D
    assert COARSE["D1/D2"] == "D"
    assert iec_ratio(10, 20, 10, 20, 0) == "T1"
    assert iec_ratio(0, 1, 0, 1, 0) == "UND"          # нулевые знаменатели -> отношения не определены
    print("Регрессионные тесты пройдены (они не подтверждают полное соответствие стандартам).")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="FinalDataSet_DGA.xlsx"); ap.add_argument("--out", default="agree_results")
    ap.add_argument("--test-boundaries", action="store_true", help="запустить регрессионные тесты и выйти")
    args = ap.parse_args()
    if args.test_boundaries:
        run_boundary_tests(); return
    out = Path(args.out); out.mkdir(exist_ok=True)
    raw = pd.read_excel(args.xlsx).rename(columns=GAS)
    audit = {"rows_raw": len(raw), "labeled_rows": int(raw["Fault"].notna().sum())}
    df = raw.dropna(subset=GASES).copy()
    audit["dropped_missing_gas"] = len(raw) - len(df)
    n0 = len(df); df = df[(df[GASES] > 0).any(axis=1) | False]; audit["dropped_all_zero"] = n0 - len(df)
    g = df.groupby(GASES, sort=False)["Fault"]
    dup_groups = g.size()[g.size() > 1]
    audit["duplicate_vector_groups"] = int(len(dup_groups))
    audit["dup_groups_with_conflicting_labels"] = int((g.nunique()[dup_groups.index] > 1).sum())
    audit["labeled_rows_before_dedup"] = int(df["Fault"].notna().sum())
    n0 = len(df); df = df.drop_duplicates(subset=GASES, keep="first").reset_index(drop=True)
    audit["labeled_rows_after_dedup"] = int(df["Fault"].notna().sum())
    audit["dropped_duplicate_gas_vectors"] = n0 - len(df); audit["rows_used"] = len(df)
    a, b = classify(df); df["duval"], df["iec"] = a, b
    df["acceptable"] = agree(a, b); df["acceptable_lenient"] = agree(a, b, True)
    df["duval_subclass_ref"] = df["duval"]            # справочно, не является согласованным диагнозом
    df["agreed_class"] = np.where(df["acceptable"], df["duval"].map(COARSE), "ПРОВЕРКА")
    df["decision"] = df["agreed_class"]

    n = len(df)
    both = (df["duval"] != "UND") & ~df["iec"].isin(["UND", "AMB"])
    kappa = cohen_kappa_score(df.loc[both, "duval"].map(COARSE), df.loc[both, "iec"].map(COARSE)) if both.sum() > 1 else np.nan
    summ = {**audit, "analysis_version": "agreement-rule-v3-D1D2-overlap", "seed": SEED, "noise_repeats_per_bound": NOISE_REPEATS, "noise_bounds": NOISE_BOUNDS,
            "duval_defined_share": float((df["duval"] != "UND").mean()),
            "iec_defined_share": float((df["iec"] == df["iec"]).mean() - df["iec"].isin(["UND", "AMB"]).mean()),
            "iec_undefined_share": float((df["iec"] == "UND").mean()), "iec_ambiguous_share": float((df["iec"] == "AMB").mean()),
            "both_defined_share": float(both.mean()),
            "coarse_agreement_among_both_defined": float(df.loc[both, "acceptable"].mean()),
            "coarse_agreement_n": int(df.loc[both, "acceptable"].sum()), "coarse_agreement_denominator": int(both.sum()),
            "coarse_agreement_wilson95": list(wilson(int(df.loc[both, "acceptable"].sum()), int(both.sum()))),
            "cohen_kappa_coarse": float(kappa),
            "hybrid_coverage_strict": float(df["acceptable"].mean()),
            "hybrid_coverage_lenient": float(df["acceptable_lenient"].mean()),
            "duval_class_counts": df["duval"].value_counts().to_dict(), "iec_class_counts": df["iec"].value_counts().to_dict()}

    pd.crosstab(df["duval"], df["iec"]).to_csv(out/"crosstab_duval_vs_iec.csv", encoding="utf-8-sig")

    # охват в зависимости от уровня газов (сумма CH4+C2H4+C2H2 и TDCG)
    df["tdcg5"] = df[GASES].sum(axis=1)
    rows = []
    for lo, hi in [(0, 10), (10, 50), (50, 200), (200, 1000), (1000, 1e12)]:
        m = (df["tdcg5"] >= lo) & (df["tdcg5"] < hi)
        if m.sum() == 0: continue
        k, nn = int(df.loc[m, "acceptable"].sum()), int(m.sum()); lo_ci, hi_ci = wilson(k, nn)
        rows.append({"tdcg5_ppm_range": f"{lo:g}-{hi:g}" if hi < 1e11 else f">={lo:g}", "n": nn,
                     "coverage_strict": k/nn, "ci95_low": lo_ci, "ci95_high": hi_ci,
                     "iec_undefined_share": float((df.loc[m, "iec"].isin(["UND", "AMB"])).mean())})
    strat = pd.DataFrame(rows); strat.to_csv(out/"coverage_by_gas_level.csv", index=False, encoding="utf-8-sig")

    # устойчивость к шуму
    rng = np.random.default_rng(SEED); noise_rows = []
    X = df[GASES].to_numpy(float)
    for bd in NOISE_BOUNDS:
        for rep in range(NOISE_REPEATS):
            Xn = X * rng.uniform(1-bd, 1+bd, size=X.shape)
            dn = pd.DataFrame(Xn, columns=GASES); an, bn = classify(dn); ok = agree(an, bn)
            ca = lambda v: np.array([COARSE.get(x, x) for x in v], dtype=object)   # UND/AMB остаются как есть
            noise_rows.append({"bound": bd, "rep": rep, "duval_changed": float((an != a).mean()), "iec_changed": float((bn != b).mean()),
                               "duval_coarse_changed": float((ca(an) != ca(a)).mean()), "iec_coarse_changed": float((ca(bn) != ca(b)).mean()),
                               "coverage": float(ok.mean()), "decision_changed": float((np.where(ok, [COARSE.get(x, "U") for x in an], "R") != np.where(df["acceptable"], [COARSE.get(x, "U") for x in a], "R")).mean())})
    nz = pd.DataFrame(noise_rows); nz.to_csv(out/"noise_repeats.csv", index=False)
    nsum = nz.groupby("bound").agg(["mean", "std"]).drop(columns=["rep"], level=0).round(4)
    nsum.columns = ["_".join(c) for c in nsum.columns]; nsum.to_csv(out/"noise_summary.csv", encoding="utf-8-sig")

    # чувствительность к входному фильтру: исходные записи сохраняются, показывается цена фильтра
    srows = []
    lab_mask = df["Fault"].notna()
    for L in [0, 1, 5, 10, 20, 50, 100]:
        dec, suff = three_tier(df, L); ns = int(suff.sum())
        acc = df["acceptable"].to_numpy() & suff
        both_s = both.to_numpy() & suff
        nf = lab_mask & (df["Fault"] == "No Fault"); fl = lab_mask & (df["Fault"] != "No Fault")
        srows.append({"min_gas_ppm_L": L, "n_sufficient": ns, "share_sufficient": ns/n,
                      "agree_share_of_sufficient": acc.sum()/ns if ns else np.nan,
                      "agree_share_of_all": acc.sum()/n,
                      "agree_among_both_defined": df["acceptable"].to_numpy()[both_s].mean() if both_s.sum() else np.nan,
                      "labeled_NoFault_given_class": int((acc & nf.to_numpy()).sum()), "labeled_NoFault_total": int(nf.sum()),
                      "labeled_Fault_given_class": int((acc & fl.to_numpy()).sum()), "labeled_Fault_total": int(fl.sum())})
    pd.DataFrame(srows).to_csv(out/"input_filter_sensitivity.csv", index=False, encoding="utf-8-sig")
    print("\nЧувствительность к входному фильтру:"); print(pd.DataFrame(srows).round(3).to_string(index=False))

    # размеченные записи — только иллюстрация
    lab = df.dropna(subset=["Fault"])[["Fault"] + GASES + ["duval", "iec", "agreed_class"]]
    lab.to_csv(out/"labeled_rows_illustration.csv", index=False, encoding="utf-8-sig")
    df.to_csv(out/"row_level_results.csv", index=False, encoding="utf-8-sig")
    json.dump(summ, open(out/"summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # рисунки
    fig, ax = plt.subplots(figsize=(7, 4.5)); s = strat
    ax.errorbar(range(len(s)), s["coverage_strict"], yerr=[s["coverage_strict"]-s["ci95_low"], s["ci95_high"]-s["coverage_strict"]], fmt="o-", capsize=4)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s["tdcg5_ppm_range"]); ax.set_ylim(0, 1)
    ax.set_xlabel("Сумма пяти газов (единицы исходного файла)"); ax.set_ylabel("Доля случаев, где методы согласны"); ax.grid(alpha=.3)
    ax.set_title("Охват правила «согласие методов» по уровню газов"); fig.tight_layout(); fig.savefig(out/"fig_coverage_by_level.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4)); g = nz.groupby("bound")[["duval_coarse_changed", "iec_coarse_changed", "decision_changed"]].mean()
    g.plot.bar(ax=ax); ax.set_ylabel("Доля изменившихся заключений"); ax.set_xlabel("Граница шума"); ax.legend(["Duval (укрупн. класс)", "IEC (укрупн. класс)", "Итог правила"])
    ax.set_title("Устойчивость к шуму"); fig.tight_layout(); fig.savefig(out/"fig_noise.png", dpi=160); plt.close(fig)
    print(json.dumps(summ, ensure_ascii=False, indent=2)); print(strat.round(3).to_string(index=False)); print(nsum.to_string())
    print(lab.to_string(index=False))

if __name__ == "__main__": main()
