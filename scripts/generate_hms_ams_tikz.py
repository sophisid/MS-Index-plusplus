#!/usr/bin/env python3.10
"""Generate a five-dataset query-time grid for HMS-Index and AMS-Index."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("synthetic", "stocks", "wind", "weather", "DuckDuckGeese")
REPRESENTATIONS = ("raw", "normalized")
KS = (1, 10, 100)
SERIES = (
    ("mass_time_ms", "MASS", "massColor", "mark=triangle*, mark options={rotate=180, fill=massColor}"),
    ("msindex_time_ms", "MS-Index", "msindexColor", "mark=*, mark options={fill=msindexColor}"),
    ("hmsindex_time_ms", "HMS-Index", "hmsColor", "mark=square*, mark options={fill=hmsColor}"),
    ("amsindex_time_ms", "AMS-Index", "amsColor", "mark=diamond*, mark options={fill=amsColor}"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def format_float(value, digits=4):
    return ("{0:." + str(digits) + "f}").format(value).rstrip("0").rstrip(".")


def dataset_label(dataset):
    return {
        "synthetic": "Synthetic",
        "stocks": "Stocks",
        "wind": "Wind",
        "weather": "Weather",
        "DuckDuckGeese": "DuckDuckGeese",
    }[dataset]


def load_rows(path):
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["representation"], int(float(row["k"])))
            grouped[key].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["noise"]))
    return grouped


def compute_y_bounds(grouped):
    values = []
    for rows in grouped.values():
        for row in rows:
            for field, _, _, _ in SERIES:
                if row.get(field) not in (None, ""):
                    values.append(float(row[field]))
    return min(values) / 1.35, max(values) * 1.35


def build_tex(grouped):
    n_cols = len(REPRESENTATIONS) * len(KS)
    n_rows = len(DATASETS)
    y_lo, y_hi = compute_y_bounds(grouped)
    lines = [
        "% Requires in preamble:",
        "% \\usepackage{pgfplots}",
        "% \\usepackage{tikz}",
        "% \\usepackage{xcolor}",
        "% \\pgfplotsset{compat=1.18}",
        "% \\usetikzlibrary{calc}",
        "% \\usepgfplotslibrary{groupplots}",
        "\\definecolor{massColor}{RGB}{244,162,97}",
        "\\definecolor{msindexColor}{RGB}{33,150,243}",
        "\\definecolor{hmsColor}{RGB}{124,179,66}",
        "\\definecolor{amsColor}{RGB}{220,68,55}",
        "\\begin{figure*}[t]",
        "\\centering",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tikzpicture}",
        "\\begin{groupplot}[",
        "  group style={group name=grp, group size=%d by %d, horizontal sep=0.15cm, vertical sep=0.5cm, xticklabels at=edge bottom, yticklabels at=edge left},"
        % (n_cols, n_rows),
        "  width=0.29\\textwidth,",
        "  height=0.18\\textwidth,",
        "  ymode=log,",
        "  ymin=%.3g, ymax=%.3g," % (y_lo, y_hi),
        "  xmin=0.0, xmax=0.9,",
        "  xtick={0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9},",
        "  xticklabel style={rotate=45, anchor=east},",
        "  tick label style={font=\\scriptsize},",
        "  label style={font=\\footnotesize},",
        "  title style={font=\\footnotesize, align=center},",
        "  legend style={font=\\scriptsize, draw=none, fill=none, legend columns=-1},",
        "  every axis plot/.append style={mark size=1.7pt},",
        "]",
    ]

    first_axis = True
    for dataset_index, dataset in enumerate(DATASETS):
        for representation in REPRESENTATIONS:
            for k in KS:
                opts = []
                if dataset_index == 0:
                    opts.append("title={$k=%s$}" % k)
                if representation == "raw" and k == 1:
                    opts.append("ylabel={Query time (ms)}")
                if dataset_index == len(DATASETS) - 1:
                    opts.append("xlabel={Noise}")
                if first_axis:
                    opts.append("legend to name=combinedlegend")
                lines.append("\\nextgroupplot" + ("[{},]".format(", ".join(opts)) if opts else ""))
                rows = grouped[(dataset, representation, k)]
                for field, label, color, marker in SERIES:
                    coords = " ".join(
                        "({},{})".format(
                            format_float(float(row["noise"]), 1),
                            format_float(float(row[field])),
                        )
                        for row in rows
                    )
                    lines.append(
                        "\\addplot+[{0}, {1}, line width=0.9pt] coordinates {{{2}}};".format(
                            color, marker, coords
                        )
                    )
                    if first_axis:
                        lines.append("\\addlegendentry{%s}" % label)
                first_axis = False

    lines.extend(
        [
            "\\end{groupplot}",
            "\\node[font=\\footnotesize\\bfseries, anchor=south] (rawhdr) at ($(grp c1r1.north west)!0.5!(grp c3r1.north east)+(0,14pt)$) {Raw};",
            "\\node[font=\\footnotesize\\bfseries, anchor=south] (normhdr) at ($(grp c4r1.north west)!0.5!(grp c6r1.north east)+(0,14pt)$) {Normalized};",
            "\\coordinate (divx) at ($(grp c3r1.east)!0.5!(grp c4r1.west)$);",
            "\\draw[gray!60, line width=0.4pt] (divx |- rawhdr.north) -- (divx |- grp c3r5.south);",
            "\\node[anchor=south] at ($(rawhdr.north)!0.5!(normhdr.north)+(0,4pt)$) {\\pgfplotslegendfromname{combinedlegend}};",
        ]
    )
    for row_index, dataset in enumerate(DATASETS, start=1):
        lines.append(
            "\\node[rotate=90, anchor=center, font=\\footnotesize\\bfseries] at ($(grp c1r%s.west)+(-13mm,0)$) {%s};"
            % (row_index, dataset_label(dataset))
        )
    lines.extend(
        [
            "\\end{tikzpicture}%",
            "}",
            "\\caption{Mean query time versus noise for the five main datasets. Each row corresponds to one dataset; columns show Raw and Normalized settings for $k\\in\\{1,10,100\\}$.}",
            "\\label{fig:hms-ams-query-time-grid}",
            "\\end{figure*}",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    grouped = load_rows(args.summary_csv)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(build_tex(grouped), encoding="utf-8")
    print(args.output_tex)


if __name__ == "__main__":
    main()
