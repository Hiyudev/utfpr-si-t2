import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from libs.input import read_data


def main():
    examples = read_data("./assets/treino_sinais_vitais_com_label.txt")

    pressao = [example.q_pa for example in examples]
    pulso = [example.pulso for example in examples]
    respiracao = [example.respiracao for example in examples]

    POINT_SCATTER = True
    SURFACE = not POINT_SCATTER

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


if __name__ == "__main__":
    main()
