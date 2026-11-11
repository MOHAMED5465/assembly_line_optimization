import pulp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def deterministic_optimization(data_dir:str, with_plots=False, plots_dir:str|None=None):
    # data loading
    equipements = pd.read_excel(data_dir, "Equipements", index_col=0)
    generic = pd.read_excel(data_dir, "Generic", index_col=0)
    operations = pd.read_excel(data_dir, "Operations", index_col="Operation")

    # equipements
    equipement_id = equipements.index.to_list()
    category = equipements['Category']
    supplier = equipements['Supplier']
    if 'Age' in equipements.columns:
        age = equipements['Age']
    speed_factor = equipements['Speed factor']
    purshasing_cost = equipements['Purchasing cost']
    operation_cost = equipements['Operation cost']
    salvage_cost = equipements['Salvage cost ']
    # generic
    takt = generic[generic.index=="Takt"]['Value'].values
    horizon = generic[generic.index=="Horizon"]['Value'].values
    lost_sale_penalty = generic[generic.index=="LostSalePenalty"]['Value'].values
    years = [i+1 for i in range(10)]
    duration_year_seconds = 24*3600*365
    # operations
    resource_group = operations['Resource Group'].unique()
    operation = operations.index.unique().to_list()
    duration_operation_year = operations.iloc[:, 1:].values

    # problem formulation
    problem = pulp.LpProblem("Assembly Line", sense=pulp.LpMinimize)

    ## variables
    X = pulp.LpVariable.dicts("X", (equipement_id, resource_group, years), cat="Integer", lowBound=0)
    Y = pulp.LpVariable.dicts("Y", equipement_id, cat="Integer", lowBound=0)
    F = pulp.LpVariable.dicts("F", (resource_group, years), cat="Continuous", lowBound=0)
    M = pulp.LpVariable.dicts("M", years, cat="Continuous", lowBound=0)

    ## objective function
    problem += (
        pulp.lpSum([Y[e] * purshasing_cost[e] for e in equipement_id]) +
        pulp.lpSum([X[e][g][y] * operation_cost[e] for e in equipement_id for g in resource_group for y in years]) -
        pulp.lpSum([Y[e] * salvage_cost[e] for e in equipement_id ]) +
        pulp.lpSum([M[y] * duration_year_seconds * lost_sale_penalty[0] for y in years])
    )

    ## constraints
    for g in resource_group:
        for year_index, y in enumerate(years):
            mask = (operations["Resource Group"] == g)
            indices = np.where(mask)[0]
            total = duration_operation_year[indices, year_index].sum()
            if total!=0:
                problem += F[g][y] == pulp.lpSum([X[e][g][y]*speed_factor[e] for e in equipement_id])/total
            else:
                problem += F[g][y] == 0
    for y in years:
        for e in equipement_id:
            problem+= pulp.lpSum([X[e][g][y] for g in resource_group]) <= Y[e]

    for g in resource_group:
        for y in years:
            problem += M[y] >= (1/takt[0]) - F[g][y]

    # solving the problem
    solver = pulp.GUROBI_CMD()
    problem.solve()
    problem.solve(solver)
    print(f"Problem status: {problem.status}")
    for var in problem.variables():
        print(f" {var.name}: {var.value()}")

    print(f"Objective function: {problem.objective.value()}")

    # plots
    if with_plots:
        sns.set_theme(style="whitegrid")
        ## number of equipments purshased
        y_data = {e: Y[e].varValue for e in equipement_id if Y[e].varValue and Y[e].varValue > 0}

        plt.figure(figsize=(10, 5))
        plt.bar(y_data.keys(), y_data.values(), color='#4C72B0')
        plt.title("Number of Equipments Purchased at the Beginning of the Cycle")
        plt.ylabel("Quantity")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "equipments_purchased.png"), dpi=150)
        plt.close()

        ## production frequency per group
        plt.figure(figsize=(12, 6))
        target_freq = 1 / takt[0]

        for g in resource_group:
            f_vals = [F[g][y].varValue for y in years]
            plt.plot(years, f_vals, marker='o', linewidth=2, label=f'Group {g}')

        plt.axhline(y=target_freq, color='#C44E52', linestyle='--', linewidth=2, label='Target Requirement (1/Takt)')
        plt.title("Evolution of Actual Production Frequency per Group")
        plt.xlabel("Years")
        plt.ylabel("Frequency (products/s)")
        plt.xticks(years)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "production_frequency.png"), dpi=150)
        plt.close()

        ## maximum deviation from production target
        m_vals = [M[y].varValue for y in years]
        plt.figure(figsize=(10, 5))
        plt.plot(years, m_vals, color="#DD5252")
        plt.title("Maximum Deviation from Production Target (Penalty)")
        plt.xlabel("Years")
        plt.ylabel("Deviation (products/s)")
        plt.xticks(years)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "maximum_deviation_from_production_target.png"), dpi=150)
        plt.close()

