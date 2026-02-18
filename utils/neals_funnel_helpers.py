import numpy as np


def f_global_z(x: np.ndarray):
    """
    Function which returns the global latent variable (z).

    Args:
        x (np.ndarray): Samples with x[0, :] = z.

    Returns:
        np.ndarray: Global latent variable (z).
    """
    z = x[0, :]

    return z


def f_global_z2(x: np.ndarray):
    """
    Function which returns the second moment of the global latent variable (z).

    Args:
        x (np.ndarray): Samples with x[0, :] = z.

    Returns:
        np.ndarray: Second moment of global latent variable (z).
    """
    z = x[0, :]

    return z**2

def f_local_squared(x: np.ndarray):
    """
    Function which returns the squared first local variable (x_1).

    Args:
        x (np.ndarray): Samples with x[1, :] = x_1.

    Returns:
        np.ndarray: Squared first local variable (x_1).
    """
    x1 = x[1, :]

    return x1**2

def f_tail_2(x: np.ndarray):
    """
    Function which returns the tail indicator function I(|z| > 2)

    Args:
        x (np.ndarray): Samples with x[0, :] = z.

    Returns:
        np.ndarray: Tail indicator function I(|z| > 2)
    """
    z = x[0, :]

    return (np.abs(z) > 2).astype(float)

def f_tail_3(x: np.ndarray):
    """
    Function which returns the tail indicator function I(|z| > 3)

    Args:
        x (np.ndarray): Samples with x[0, :] = z.

    Returns:
        np.ndarray: Tail indicator function I(|z| > 3)
    """
    z = x[0, :]

    return (np.abs(z) > 3).astype(float)

def make_results_table(dim: int, metrics: list, results: dict):
    """
    Function which prints a results table for the Neal's funnel example.

    Args:
        dim (int): Dimensionality of the distribution
        metrics (list): List of tuples containing the label and the number of decimals to display.
        results (dict): Dictionary containing the results for each method. Keys are the method names, 
        values are dictionaries with metric labels as keys and tuples of (value, standard_error or None)
        as values, where None indicates that the standard error is not available.
        
        Example:
        {"Method 1": {"Metric 1": (value1, se_value1), "Metric 2": (value2, None)},
         "Method 2": {"Metric 1": (value3, se_value3), "Metric 2": (value4, None)}}
    """

    def fmt_value(x, decimals):
        if abs(x) < 0.5 * (10 ** -decimals):
            return "0"
        return f"{x:.{decimals}f}"

    def fmt_standard(x, decimals):
        if x == 0:
            return "0"
        return f"{x:.{decimals}e}"

    VALUE_DECIMALS = 3
    SE_DECIMALS = 2

    # Header
    header = [f"Method (d={dim})"] + [m[0] for m in metrics]

    # Rows
    rows = []
    for method, method_data in results.items():
        row = [method]
        for label, decimals in metrics:
            value, se = method_data[label]
            if se is None:
                cell = fmt_value(value, VALUE_DECIMALS)
            else:
                value_str = fmt_value(value, VALUE_DECIMALS)
                se_str = fmt_standard(se, SE_DECIMALS)
                cell = f"{value_str} ± {se_str}"

            row.append(cell)

        rows.append(row)

    # Column widths
    cols = list(zip(header, *rows))
    col_widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row):
        out = []
        for i, cell in enumerate(row):
            if i == 0:
                out.append(str(cell).ljust(col_widths[i]))
            else:
                out.append(str(cell).rjust(col_widths[i]))
        return "| " + " | ".join(out) + " |"

    sep = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [
        fmt_row(header),
        sep,
        *[fmt_row(r) for r in rows]
    ]

    print("\n".join(lines))