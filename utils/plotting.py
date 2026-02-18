import numpy as np
import matplotlib.pyplot as plt

def plot_2dim_scatter(scatters: list, save_path: str | None = None):
    """
    Plot scatter plots for an arbitrary number of scatters.

    Args:
        scatters (list): List of dictionaries, each containing the data and label of a scatter plot.
        Each dict must have keys 'data' (np.ndarray) and 'label' (str). Optional key 'color' (or 'colour') 
        overrides the default color for that scatter. Example:
        [{"data": np.random.randn(2, 100), "label": "Target"}, {"data2": np.random.randn(2, 100), "label2": "Pushforward", "color": "red"}, ...]
        save_path (str | None): Path to save the figure. If None, the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterator over the default matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_iter = iter(colors)

    for scatter in scatters:
        kwargs = dict(
            label=scatter["label"],
            alpha=0.5,
            linewidths=3,
        )

        if "color" in scatter:      
            kwargs["color"] = scatter["color"]
            next(color_iter, None)
        elif "colour" in scatter:
            kwargs["color"] = scatter["colour"]
            next(color_iter, None)
        else:
            c = next(color_iter, None)
            if c is None:
                color_iter = iter(colors)
                c = next(color_iter)
            kwargs["color"] = c

        ax.scatter(scatter["data"][0, :], scatter["data"][1, :], **kwargs)

    plt.legend(fontsize=18, loc='lower left')
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel(r'$x_0$',fontsize=25)
    plt.ylabel(r'$x_1$',fontsize=25)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def plot_radial_sortings(curves: list, type: str = "isotropic", save_path: str | None = None):
    """
    Plot radial sortings for an isotropic or anisotropic distribution.

    Args:
        curves (list): List of dictionaries, each containing the data and label of a curve.
        Each dict must have keys 'data' (np.ndarray) and 'label' (str). Optional key 'color' (or 'colour') 
        overrides the default color for that curve. Optional key 'linestyle' overrides the default (solid line '-') 
        for that curve. Example:
        [{"data": r_target, "label": "Target"}, {"data": r_pushforward, "label": r"\\texttt{radVI}", "color": "red", "linestyle": "--"}, ...]
        type (str): Type of distribution. Can be "isotropic" or "anisotropic". Default is "isotropic".
        save_path (str | None): Path to save the figure. If None, the figure is not saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Iterator over the default matplotlib color cycle
    colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    color_iter = iter(colors)

    for curve in curves:
        if type == "isotropic":
            kwargs = dict(label=curve["label"], alpha=0.5, linestyle=curve.get("linestyle", "-"), linewidth=3.5, marker=".",  markersize=3)
        
        elif type == "anisotropic":
            kwargs = dict(label=curve["label"], linestyle=curve.get("linestyle", "-"), linewidth=3.5)

        # If user provided a color/colour, use it BUT still advance the cycle by consuming one
        if "color" in curve:
            kwargs["color"] = curve["color"]
            next(color_iter, None)  # advance by 1 safely
        elif "colour" in curve:
            kwargs["color"] = curve["colour"]
            next(color_iter, None)
        else:
            # No explicit color -> take the next default
            c = next(color_iter, None)
            if c is None:
                # cycle exhausted; restart
                color_iter = iter(colors)
                c = next(color_iter)
            kwargs["color"] = c

        ax.plot(np.sort(curve["data"]), **kwargs)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Sorted $r$", fontsize=25)
    plt.ylabel("Sorted radial pushforward", fontsize=25)
    plt.legend(markerscale=5, fontsize=18)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()