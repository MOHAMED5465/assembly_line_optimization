from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns
from scipy.stats import truncnorm


def generate_scenarios(t_base, n_scenarios, distribution="uniform", sigma_ratio=0.2, rng=None):
    """Return an array of shape (scenarios, operations, years).

    Uniform durations are sampled between t and 2t. Truncated-normal
    durations have mean parameter t, standard deviation sigma_ratio*t,
    and bounds [t, 2t], matching the notebook's assumptions.
    """
    t_base = np.asarray(t_base, dtype=float)
    if t_base.ndim != 2 or not np.isfinite(t_base).all() or (t_base < 0).any():
        raise ValueError("t_base must be a finite nonnegative 2D array")
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be positive")
    if rng is None:
        rng = np.random.default_rng()
    shape = (n_scenarios, *t_base.shape)
    if distribution == "uniform":
        return rng.uniform(t_base, 2 * t_base, size=shape)
    if distribution == "truncated_normal":
        if sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be positive")
        # keep zero-duration operations unchanged
        out = np.zeros(shape, dtype=float)
        positive = t_base > 0
        if positive.any():
            means = t_base[positive]
            scales = sigma_ratio * means
            a = np.zeros_like(means)
            b = means / scales
            out[:, positive] = truncnorm.rvs(
                a=a, b=b, loc=means, scale=scales,
                size=(n_scenarios, len(means)), random_state=rng
            )
        return out
    raise ValueError("distribution must be 'uniform' or 'truncated_normal'")


def _load_data(data_dir):
    equipements = pd.read_excel(data_dir, sheet_name="Equipements", index_col=0)
    generic = pd.read_excel(data_dir, sheet_name="Generic", index_col=0)
    operations = pd.read_excel(data_dir, sheet_name="Operations", index_col="Operation")
    durations = operations.iloc[:, 1:].to_numpy(dtype=float)
    years = list(range(1, durations.shape[1] + 1))
    if not years:
        raise ValueError("No yearly duration columns found in Operations")
    for key in ("Takt", "LostSalePenalty"):
        if key not in generic.index:
            raise ValueError(f"Missing '{key}' in Generic sheet")
    takt = float(generic.loc["Takt", "Value"])
    if takt <= 0:
        raise ValueError("Takt must be positive")
    return {
        "equipements": equipements,
        "operations": operations,
        "durations": durations,
        "years": years,
        "equipement_id": equipements.index.to_list(),
        "resource_group": operations["Resource Group"].unique().tolist(),
        "speed_factor": equipements["Speed factor"],
        "purchasing_cost": equipements["Purchasing cost"],
        "operation_cost": equipements["Operation cost"],
        "salvage_cost": equipements["Salvage cost "],
        "takt": takt,
        "lost_sale_penalty": float(generic.loc["LostSalePenalty", "Value"]),
        "duration_year_seconds": 24 * 3600 * 365,
    }


def _solve_model(data, durations, solver=None, fixed_y=None):
    """Solve the notebook's two-stage MILP for supplied scenarios."""
    durations = np.asarray(durations, dtype=float)
    base_shape = data["durations"].shape
    if durations.ndim != 3 or durations.shape[1:] != base_shape:
        raise ValueError(f"durations shape must be (N, {base_shape[0]}, {base_shape[1]})")
    if not np.isfinite(durations).all() or (durations < 0).any():
        raise ValueError("Scenario durations must be finite and nonnegative")

    equipement_id = data["equipement_id"]
    resource_group = data["resource_group"]
    years = data["years"]
    scenarios = range(durations.shape[0])
    n = durations.shape[0]
    if not n:
        raise ValueError("At least one scenario is required")

    problem = pulp.LpProblem("Assembly_Line_Stochastic", pulp.LpMinimize)
    Y = pulp.LpVariable.dicts("Y", equipement_id, lowBound=0, cat="Integer")
    if fixed_y is not None:
        for e in equipement_id:
            if e not in fixed_y:
                raise ValueError(f"Missing fixed equipment quantity: {e}")
            quantity = float(fixed_y[e])
            if not np.isfinite(quantity) or quantity < 0 or not np.isclose(quantity, round(quantity)):
                raise ValueError(f"Invalid fixed equipment quantity for {e}: {quantity}")
            problem += Y[e] == round(quantity)

    X = pulp.LpVariable.dicts("X", (equipement_id, resource_group, years, scenarios), lowBound=0, cat="Integer")
    F = pulp.LpVariable.dicts("F", (resource_group, years, scenarios), lowBound=0)
    M = pulp.LpVariable.dicts("M", (years, scenarios), lowBound=0)

    first_stage = pulp.lpSum(
        Y[e] * (data["purchasing_cost"][e] - data["salvage_cost"][e])
        for e in equipement_id
    )
    second_stage_operation = pulp.lpSum(
        X[e][g][yr][s] * data["operation_cost"][e] / n
        for e in equipement_id for g in resource_group
        for yr in years for s in scenarios
    )
    second_stage_shortfall = pulp.lpSum(
        M[yr][s] * data["duration_year_seconds"] * data["lost_sale_penalty"] / n
        for yr in years for s in scenarios
    )
    problem += first_stage + second_stage_operation + second_stage_shortfall

    groups = data["operations"]["Resource Group"].to_numpy()
    indices_by_group = {g: np.flatnonzero(groups == g) for g in resource_group}
    for g in resource_group:
        indices = indices_by_group[g]
        for j, yr in enumerate(years):
            for s in scenarios:
                total = float(durations[s, indices, j].sum())
                if total > 0:
                    problem += F[g][yr][s] == (
                        pulp.lpSum(X[e][g][yr][s] * data["speed_factor"][e]
                                   for e in equipement_id) / total
                    )
                else:
                    problem += F[g][yr][s] == 0

    for e in equipement_id:
        for yr in years:
            for s in scenarios:
                problem += pulp.lpSum(X[e][g][yr][s] for g in resource_group) <= Y[e]
    for g in resource_group:
        for yr in years:
            for s in scenarios:
                problem += M[yr][s] >= 1 / data["takt"] - F[g][yr][s]

    problem.solve(solver if solver is not None else pulp.PULP_CBC_CMD(msg=False))
    if problem.status != pulp.LpStatusOptimal:
        raise RuntimeError(f"Optimization failed: {pulp.LpStatus[problem.status]}")
    return {
        "model": problem, "Y": Y, "X": X, "F": F, "M": M,
        "scenarios": scenarios, "objective": pulp.value(problem.objective),
    }


def _extract_frequency_data(data, solution):
    return pd.DataFrame([
        {"group": g, "year": yr, "scenario": s,
         "frequency": pulp.value(solution["F"][g][yr][s])}
        for g in data["resource_group"] for yr in data["years"]
        for s in solution["scenarios"]
    ])


def _extract_cost_data(data, solutions):
    rows = []
    for approach, solution in solutions.items():
        n = len(solution["scenarios"])
        first = sum(
            pulp.value(solution["Y"][e]) *
            (data["purchasing_cost"][e] - data["salvage_cost"][e])
            for e in data["equipement_id"]
        )
        operations = sum(
            pulp.value(solution["X"][e][g][yr][s]) * data["operation_cost"][e]
            for e in data["equipement_id"] for g in data["resource_group"]
            for yr in data["years"] for s in solution["scenarios"]
        ) / n
        shortfall = sum(
            pulp.value(solution["M"][yr][s]) * data["duration_year_seconds"] * data["lost_sale_penalty"]
            for yr in data["years"] for s in solution["scenarios"]
        ) / n
        second = operations + shortfall
        if not np.isclose(first + second, solution["objective"], rtol=1e-5, atol=1e-4):
            raise RuntimeError(f"Cost decomposition mismatch for {approach}")
        rows.extend([
            {"approach": approach, "cost_type": "1st stage (capex)", "value": first},
            {"approach": approach, "cost_type": "expected 2nd stage", "value": second},
        ])
    return pd.DataFrame(rows)


def _extract_convergence_data(data, scenario_counts, n_replications, sigma_ratio, rng, solver):
    if n_replications < 1:
        raise ValueError("n_replications must be >= 1")
    rows = []
    for distribution in ("uniform", "truncated_normal"):
        for n in scenario_counts:
            for rep in range(n_replications):
                samples = generate_scenarios(data["durations"], n, distribution, sigma_ratio, rng)
                solution = _solve_model(data, samples, solver)
                rows.append({
                    "n_scenarios": n,
                    "distribution": "uniform" if distribution == "uniform" else "truncated normal",
                    "replication": rep,
                    "objective_value": solution["objective"],
                })
    return pd.DataFrame(rows)


def _expected_durations(t_base, distribution, sigma_ratio):
    t_base = np.asarray(t_base, dtype=float)
    if distribution == "uniform":
        return 1.5 * t_base
    if distribution != "truncated_normal":
        raise ValueError("Unknown distribution")
    if sigma_ratio <= 0:
        raise ValueError("sigma_ratio must be positive")
    expected = np.zeros_like(t_base)
    positive = t_base > 0
    if positive.any():
        means = t_base[positive]
        scales = sigma_ratio * means
        expected[positive] = truncnorm.mean(
            a=np.zeros_like(means), b=means / scales,
            loc=means, scale=scales,
        )
    return expected


def _evaluate_stochastic_value(data, samples, distribution, sigma_ratio, solver, rp_solution=None):
    rp_solution = rp_solution or _solve_model(data, samples, solver)
    expected = _expected_durations(data["durations"], distribution, sigma_ratio)
    deterministic_solution = _solve_model(data, expected[np.newaxis, :, :], solver)
    fixed_y = {e: round(pulp.value(deterministic_solution["Y"][e])) for e in data["equipement_id"]}
    eev_solution = _solve_model(data, samples, solver, fixed_y=fixed_y)
    rp = float(rp_solution["objective"])
    eev = float(eev_solution["objective"])
    vss = eev - rp
    tolerance = 1e-5 * max(1.0, abs(rp), abs(eev))
    if vss < -tolerance:
        raise RuntimeError(f"EEV < RP for {distribution}: verify solver and model")
    return {
        "RP": rp,
        "deterministic_cost": float(deterministic_solution["objective"]),
        "EEV": eev,
        "VSS": max(0.0, vss),
        "rp_solution": rp_solution,
        "deterministic_solution": deterministic_solution,
        "eev_solution": eev_solution,
        "fixed_y_deterministic": fixed_y,
    }


def _save_plots(data, frequency, costs, convergence, metrics, plots_dir):
    sns.set_theme(style="whitegrid")
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: production frequency
    g = sns.catplot(
        data=frequency, x="year", y="frequency", col="group", col_wrap=2,
        kind="box", sharey=False, height=4, aspect=1.2, color="lightblue",
    )
    for ax in g.axes.flat:
        ax.axhline(y=1 / data["takt"], color="#c44e52", linestyle="--", linewidth=2)
    g.set_axis_labels("years", "frequency (products/s)")
    g.set_titles("group: {col_name}")
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle("stochastic variability of production frequency by resource group")
    g.savefig(plots_dir / "production_frequency.png", dpi=150)
    plt.close(g.fig)

    # Plot 2: objective cost components
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=costs, x="approach", y="value", hue="cost_type",
                palette=["#4c72b0", "#dd8452"], ax=ax)
    totals = costs.groupby("approach", as_index=False, sort=False)["value"].sum()
    sns.scatterplot(data=totals, x="approach", y="value", color="black",
                    marker="D", s=90, label="total cost", zorder=10, ax=ax)
    ax.set_title("first-stage and expected second-stage costs")
    ax.set_ylabel("cost (monetary unit)")
    ax.set_xlabel("optimization model")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(title="objective component")
    fig.tight_layout()
    fig.savefig(plots_dir / "cost_comparison.png", dpi=150)
    plt.close(fig)

    # Plot 3: in-sample SAA convergence
    if convergence is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=convergence, x="n_scenarios", y="objective_value",
                     hue="distribution", marker="o", linewidth=2,
                     palette=["#55a868", "#c44e52"], ax=ax)
        for distribution, color in (("uniform", "#2e7d32"), ("truncated_normal", "#9a303a")):
            ax.axhline(metrics[distribution]["deterministic_cost"], color=color, linestyle="--",
                       linewidth=1.5, label=f"deterministic solution ({distribution.replace('_', ' ')})")
        ax.set_title("in-sample SAA objective vs number of scenarios")
        ax.set_xlabel("number of scenarios (n)")
        ax.set_ylabel("optimal objective function value")
        ax.set_xscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / "saa_convergence.png", dpi=150)
        plt.close(fig)

    # Plot 4: RP, EEV and VSS
    labels = ["uniform", "truncated_normal"]
    x = np.arange(len(labels))
    width = 0.32
    rp_values = [metrics[k]["RP"] for k in labels]
    eev_values = [metrics[k]["EEV"] for k in labels]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width / 2, rp_values, width, label="RP", color="#4c72b0")
    ax.bar(x + width / 2, eev_values, width, label="EEV", color="#dd8452")
    for i, key in enumerate(labels):
        top = max(rp_values[i], eev_values[i])
        ax.annotate(f"VSS = {metrics[key]['VSS']:,.2f}",
                    xy=(x[i], top), xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, ["uniform", "truncated normal"])
    ax.set_ylabel("expected cost (monetary unit)")
    ax.set_title("value of the stochastic solution (VSS = EEV - RP)")
    ax.legend()
    ax.margins(y=0.17)
    fig.tight_layout()
    fig.savefig(plots_dir / "rp_eev_vss.png", dpi=150)
    plt.close(fig)


def stochastic_optimization(
    data_dir: str,
    with_plots: bool = False,
    plots_dir: str | None = None,
    n_scenarios: int = 10,
    distribution: str = "uniform",
    sigma_ratio: float = 0.2,
    seed: int | None = 42,
    run_convergence: bool = False,
    scenario_counts: tuple[int, ...] = (5, 10, 15, 20),
    n_replications: int = 1,
    solver=None,
):
    """Solve the stochastic model and calculate RP, EEV and VSS."""
    if distribution not in ("uniform", "truncated_normal"):
        raise ValueError("Unknown distribution")
    if with_plots and plots_dir is None:
        raise ValueError("Provide plots_dir when with_plots=True")

    data = _load_data(data_dir)
    rng = np.random.default_rng(seed)
    sampled = generate_scenarios(data["durations"], n_scenarios, distribution, sigma_ratio, rng)
    rp_solution = _solve_model(data, sampled, solver)
    selected_metrics = _evaluate_stochastic_value(
        data, sampled, distribution, sigma_ratio, solver, rp_solution=rp_solution
    )
    result = dict(rp_solution)
    result["metrics"] = {
        k: selected_metrics[k] for k in ("RP", "EEV", "VSS")
    }
    print(f"problem status: {pulp.LpStatus[result['model'].status]}")
    print(f"distribution: {distribution}")
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.4f}")

    if with_plots:
        metrics = {distribution: selected_metrics}
        for other in ("uniform", "truncated_normal"):
            if other not in metrics:
                other_samples = generate_scenarios(
                    data["durations"], n_scenarios, other, sigma_ratio, rng
                )
                metrics[other] = _evaluate_stochastic_value(
                    data, other_samples, other, sigma_ratio, solver
                )
        solutions = {
            f"deterministic solution ({key.replace('_', ' ')})": metrics[key]["deterministic_solution"]
            for key in ("uniform", "truncated_normal")
        }
        solutions.update({
            f"RP ({key.replace('_', ' ')})": metrics[key]["rp_solution"]
            for key in ("uniform", "truncated_normal")
        })
        frequency = _extract_frequency_data(data, rp_solution)
        costs = _extract_cost_data(data, solutions)
        convergence = _extract_convergence_data(
            data, scenario_counts, n_replications, sigma_ratio, rng, solver
        ) if run_convergence else None
        _save_plots(data, frequency, costs, convergence, metrics, plots_dir)
        result["distribution_metrics"] = {
            name: {k: values[k] for k in ("RP", "EEV", "VSS")}
            for name, values in metrics.items()
        }

    return result
