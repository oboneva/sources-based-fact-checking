from datetime import datetime

from matplotlib import pyplot as plt


def save_conf_matrix(disp, model_name: str, top_n_domains=None):
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)

    plot_title = (
        f"{model_name}, Top {top_n_domains} domains" if top_n_domains else model_name
    )

    plt.title(plot_title)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    filename = (
        f"conf_matrix_{model_name.lower()}_top_{top_n_domains}_{timestamp}.png"
        if top_n_domains
        else f"conf_matrix_{model_name.lower()}_{timestamp}.png"
    )

    plt.savefig(filename, pad_inches=5, dpi=300)
