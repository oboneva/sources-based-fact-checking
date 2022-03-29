from datetime import datetime

from matplotlib import pyplot as plt


def save_conf_matrix(disp, model_name: str, top_n_domains: int):
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title(f"{model_name}, Top {top_n_domains} domains")
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    plt.savefig(
        f"conf_matrix_{model_name.lower()}_top_{top_n_domains}_{timestamp}.png",
        pad_inches=5,
        dpi=300,
    )
