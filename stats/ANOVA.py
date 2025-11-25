import scipy.stats as stats
import numpy as np
import pandas as pd
from collections import defaultdict
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def one_way_anova(data,factor_attr, value_attr,alpha=0.05):
    grouped = {}  #

    # 1. Grouped by factor
    for ans in data:
        factor = getattr(ans, factor_attr)
        value = getattr(ans, value_attr)

        if value is None:
            continue

        try:
            v = float(value)
        except (TypeError, ValueError):
            continue


        if factor not in grouped:
            grouped[factor] = []
        grouped[factor].append(v)


    if len(grouped) < 2:
        raise ValueError("Need at least two groups for one-way ANOVA.")

    group_names = list(grouped.keys())
    group_values = [grouped[name] for name in group_names]


    for name, values in zip(group_names, group_values):
        if len(values) < 2:
            raise ValueError(f"Group '{name}' has fewer than 2 observations.")


    F, p = stats.f_oneway(*group_values)


    groups_info = {}
    for name, vals in grouped.items():
        n = len(vals)
        mean_val = sum(vals) / n if n > 0 else float("nan")
        groups_info[name] = {
            "values": vals,
            "n": n,
            "mean": mean_val,
        }

    return {
        "F": F,
        "p": p,
        "significant": p < alpha,
        "alpha": alpha,
        "groups": groups_info,
    }


def multi_way_anova(
    data,
    factor_attrs,
    value_attr,
    alpha=0.05,
    interaction="full",  # "none" and "full"
):
    if not factor_attrs or len(factor_attrs) < 1:
        raise ValueError("factor_attrs need at least one factor")


    rows = {
        "y": [],
    }
    for f in factor_attrs:
        rows[f] = []

    for ans in data:
        value = getattr(ans, value_attr, None)
        if value is None:
            continue

        try:
            y = float(value)
        except (TypeError, ValueError):
            continue

        rows["y"].append(y)
        for f in factor_attrs:
            rows[f].append(getattr(ans, f))

    if len(rows["y"]) < 3:
        raise ValueError("Not enough valid observations for ANOVA.")

    df = pd.DataFrame(rows)

    # 2.  formula
    # main: C(f1) + C(f2) + ...
    if interaction == "none":
        rhs_terms = [f"C({f})" for f in factor_attrs]
        rhs = " + ".join(rhs_terms)
    elif interaction == "full":
        # C(f1)*C(f2)*C(f3) ALL
        rhs_terms = [f"C({f})" for f in factor_attrs]
        rhs = " * ".join(rhs_terms)
    else:
        raise ValueError("interaction only have 'none' and 'full'")

    formula = f"y ~ {rhs}"


    model = smf.ols(formula, data=df).fit()
    anova_table = anova_lm(model, typ=2)  # Type II ANOVA


    effects = {}

    for row_name in anova_table.index:
        if row_name == "Residual":
            continue

        row = anova_table.loc[row_name]
        ss = float(row["sum_sq"])
        df_effect = float(row["df"])
        F = float(row["F"])
        p = float(row["PR(>F)"])


        pretty_name = row_name
        pretty_name = pretty_name.replace("C(", "").replace(")", "")
        pretty_name = pretty_name.replace(":", "*")

        effects[pretty_name] = {
            "ss": ss,
            "df": df_effect,
            "F": F,
            "p": p,
            "significant": p < alpha,
        }

    return {
        "alpha": alpha,
        "formula": formula,
        "effects": effects,
        "anova_table": anova_table,
    }
