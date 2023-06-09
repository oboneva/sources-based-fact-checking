import statistics


def format_results(results: str) -> str:
    lines = results.split("\n")

    ordered_lines = []

    for line in lines:
        metrics = line.split("	")
        new_order = [
            metrics[3],
            metrics[4],
            metrics[5],
            metrics[1],
            metrics[0],
            metrics[2],
        ]

        ordered_line = " &        ".join(new_order)
        ordered_lines.append(ordered_line)

    new_order = [
        ordered_lines[0],
        ordered_lines[2],
        ordered_lines[4],
        ordered_lines[6],
        ordered_lines[8],
        ordered_lines[1],
        ordered_lines[3],
        ordered_lines[5],
        ordered_lines[7],
        ordered_lines[9],
    ]

    return "       \\\ \hdashline\n".join(new_order) + "       \\\ \hdashline\n"


def compute_mean_std(values):
    std = statistics.stdev(values)
    m = statistics.mean(values)

    print("mean", m)
    print("std", std)


def main():
    # results = """ ... """
    # print(format_results(results))
    pass


if __name__ == "__main__":
    main()
