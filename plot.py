import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from libs.input import read_data
from libs.data import generate_discretized_data, DiscretizationMethod


def plot_examples_data():
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")

    pressao = [example.q_pa for example in examples]
    pulso = [example.pulso for example in examples]
    respiracao = [example.respiracao for example in examples]

    for GRAVIDADE in [True, False]:
        if GRAVIDADE:
            analysis = [example.gravidade for example in examples]
            cmap = "viridis"
            color_label = "Gravidade"
        else:
            analysis = [example.rotulo for example in examples]

            # Defina as classes e cores desejadas
            classes = sorted(set(analysis))
            colors = plt.cm.tab10.colors[: len(classes)]
            cmap = ListedColormap(colors)
            color_label = "Rotulo"

        # Plotting
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(111, projection="3d")

        scatter = ax.scatter(
            pressao, pulso, respiracao, c=analysis, cmap=cmap, marker="o"
        )
        if not GRAVIDADE:
            # Cria uma legenda personalizada para as classes
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[i],
                    markersize=10,
                    label=str(cls),
                )
                for i, cls in enumerate(classes)
            ]
            ax.legend(handles=handles, title="Classe")
        else:
            plt.colorbar(scatter, label=color_label)

        ax.set_xlabel("Pressão")
        ax.set_ylabel("Pulso")
        ax.set_zlabel("Respiração")

        plt.title("Examples data Visualization")

        file_name = "4d_plot_gravidade.png" if GRAVIDADE else "4d_plot_rotulo.png"
        file_dir = "./output/"

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        plt.savefig(file_dir + file_name, dpi=300)
        plt.show()


def plot_discretization_comparasion():
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")

    # Histograms
    file_dir = "./output/"

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    attributes = ["q_pa", "pulso", "respiracao"]
    methods = {
        "Otsu": DiscretizationMethod.OTSU,
        "Percentil": DiscretizationMethod.PERCENTIL,
        "Normal": DiscretizationMethod.NORMAL,
    }
    group_range = range(2, 6)

    results = {
        attr: {
            method: {"groups": [], "variances": [], "coef_vars": []}
            for method in methods
        }
        for attr in attributes
    }

    for attr in attributes:
        data = [getattr(example, attr) for example in examples]
        if np.min(data) < 0:
            data = np.array(data) - np.min(data)
        reshaped_data = np.array(data).reshape(-1, 1)

        for method_str, method in methods.items():
            for groups in group_range:
                try:
                    groups_data, _, _ = generate_discretized_data(
                        reshaped_data,
                        method=method,
                        parameters={"groups": groups},
                    )
                except Exception:
                    print(
                        f"Error generating data for {attr} with method {method_str} and groups {groups}. Skipping."
                    )
                    continue

                # Collect variance and coef_var for each group
                group_variances = []
                group_coef_vars = []
                for group in groups_data:
                    group_values = [
                        (
                            value.item()
                            if isinstance(value, np.ndarray) and value.size == 1
                            else value
                        )
                        for value in group
                    ]
                    if len(group_values) == 0:
                        group_variances.append(float("nan"))
                        group_coef_vars.append(float("nan"))
                        continue
                    variance = np.var(group_values)
                    mean = np.mean(group_values)
                    coef_var = (
                        (np.std(group_values) / mean) * 100
                        if mean != 0
                        else float("nan")
                    )
                    group_variances.append(variance)
                    group_coef_vars.append(coef_var)

                # Save average variance and coef_var for this group count
                avg_variance = np.nanmean(group_variances)
                avg_coef_var = np.nanmean(group_coef_vars)
                results[attr][method_str]["groups"].append(groups)
                results[attr][method_str]["variances"].append(avg_variance)
                results[attr][method_str]["coef_vars"].append(avg_coef_var)

    # Plot comparison graphs for each attribute
    file_dir = "./output/"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    for attr in attributes:
        plt.figure(figsize=(12, 6))
        for method_str in methods:
            plt.plot(
                results[attr][method_str]["groups"],
                results[attr][method_str]["variances"],
                marker="o",
                label=f"{method_str} - Variance",
            )
        plt.title(f"Average Variance vs Groups for {attr}")
        plt.xlabel("Number of Groups")
        plt.ylabel("Average Variance")
        plt.legend()
        plt.savefig(file_dir + f"variance_comparison_{attr}.png", dpi=300)
        plt.show()

        plt.figure(figsize=(12, 6))
        for method_str in methods:
            plt.plot(
                results[attr][method_str]["groups"],
                results[attr][method_str]["coef_vars"],
                marker="o",
                label=f"{method_str} - Coef. Var.",
            )
        plt.title(f"Average Coefficient of Variation vs Groups for {attr}")
        plt.xlabel("Number of Groups")
        plt.ylabel("Average Coefficient of Variation (%)")
        plt.legend()
        plt.savefig(file_dir + f"coefvar_comparison_{attr}.png", dpi=300)
        plt.show()


def main():
    plot_discretization_comparasion()


if __name__ == "__main__":
    main()
