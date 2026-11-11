from src.optimization.deterministic_optimization import deterministic_optimization
from src.optimization.stochastic_optimization import stochastic_optimization
import pulp
data_dir = r"data\ToyCar_StudentProject.xlsx"
solver = pulp.GUROBI_CMD(msg=False)
stochastic_optimization(data_dir, with_plots=True, plots_dir="plots/stochastic", solver=solver, 
                        scenario_counts=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80), 
                        run_convergence=True)
