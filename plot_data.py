from typing import Dict, List, Any, Tuple
from matplotlib import _preprocess_data, pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import os
import json
import re
from pydantic.v1.utils import deep_update
from numpy.core.multiarray import ndarray

filetype = "pdf"
outputFolder = ""
inputFolder = "data"


def data_preprocessor(folder: str):
    for fname in os.listdir(folder):
        if not fname.endswith(".dat"):
            continue
        with open(folder + "/" + fname, "r") as f:
            with open(folder + "/" + fname.replace(".dat", ".csv"), "w") as nf:
                for line in f:
                    l = line.replace(",", ".").replace(" 	", ",")
                    nf.write(l)


def get_data(filename: str) -> np.ndarray:
    data = np.loadtxt(inputFolder + "/" + filename, delimiter=",")
    return data


def numdiff(data: ndarray, col: int, dcol: int) -> ndarray:
    return np.gradient(data[:, col], data[:, dcol])


def numint(data: ndarray, col: int, dcol: int) -> ndarray:
    return np.trapz(data[:, col], data[:, dcol])


def sort_subplots(subplots: Dict, elements: Dict):
    maxSp: List[int] = [0, 0]
    for key in subplots.keys():
        if int(key.split(",")[0]) > maxSp[0]:
            maxSp[0] = int(key.split(",")[0])
        if int(key.split(",")[1]) > maxSp[1]:
            maxSp[1] = int(key.split(",")[1])
    for key in subplots.keys():
        subplots[key].update(
            {
                "plots": [],
                "rep": (
                    maxSp[0],
                    maxSp[1],
                    int(key.split(",")[1]) + maxSp[1] * (int(key.split(",")[0]) - 1),
                ),
            }
        )
    for key, element in elements.items():
        subplots[element["subplot"]]["plots"].append(element)
    return subplots


def plot_subplot(
    fig: Figure, subplot: Dict, data: Dict, grid: bool, multsrc: bool
) -> Figure:
    nrows, ncols, index = subplot["rep"]
    labels: Any = []
    axes = {"y_1": fig.add_subplot(nrows, ncols, index)}
    if "y_2" in subplot:
        axes.update({"y_2": axes["y_1"].twinx()})
    for axn, ax in axes.items():  # newshit
        if "log" in subplot[axn]:
            ax.set_yscale("log", base=subplot[axn]["log"])
    if "log" in subplot["x_1"]:
        axes["y_1"].set_xscale("log", base=subplot["x_1"]["log"])  # /newshit
    for plot in subplot["plots"]:
        if multsrc:
            x = data[plot["src"]][plot["xcol"]]
            y = data[plot["src"]][plot["ycol"]]
        else:
            x = data[plot["xcol"]]
            y = data[plot["ycol"]]
        (tmp,) = axes[plot["axis"]].plot(
            x,
            y,
            marker=plot["marker"],
            label=plot["label"],
            color=plot["color"],
            linestyle=plot["linestyle"],
            linewidth=plot["linewidth"],
        )
        labels.append(tmp)
    if subplot["legend"]:
        axes["y_1"].legend(
            labels,
            [axisForLabels_.get_label() for axisForLabels_ in labels],
            loc=subplot["legend_loc"],
            fontsize="small",
        )
    if grid:
        axes["y_1"].grid()
    axes["y_1"].set_xlabel(subplot["x_1"]["label"])
    axes["y_1"].set_ylabel(subplot["y_1"]["label"])
    axes["y_1"].set_xlim(
        left=subplot["x_1"]["limits"][0], right=subplot["x_1"]["limits"][1]
    )
    axes["y_1"].set_ylim(
        bottom=subplot["y_1"]["limits"][0], top=subplot["y_1"]["limits"][1]
    )
    if "y_2" in axes:
        axes["y_2"].set_ylim(
            bottom=subplot["y_2"]["limits"][0], top=subplot["y_2"]["limits"][1]
        )
        axes["y_2"].set_ylabel(subplot["y_2"]["label"])
    return fig


def do_stuff(name: str, opts: Dict, data: Dict):
    # for plot, cfg in opts.items():
    cfg = opts
    if "src" not in opts["data"]:
        orgs: List[str] = []
    elif opts["data"]["src"] is None:
        orgs = []
        opts["data"].pop("src")
    elif isinstance(opts["data"]["src"], str):
        regex = re.compile(opts["data"]["src"])
        orgs = [org for org in data.keys() if re.match(regex, org)]
        opts["data"].pop("src")
    elif isinstance(opts["data"]["src"], list):
        orgs = opts["data"]["src"]
        opts["data"].pop("src")
    else:
        raise ValueError(
            f"Value \"{opts['data']['src']}\" in data of plots is not valid"
        )
    subplots = sort_subplots(cfg["subplots"], cfg["data"])
    if orgs == []:
        fig = plt.figure(
            dpi=opts["dpi"], figsize=tuple([e / 2.54 for e in opts["figsize"]])
        )
        for _, subplot in subplots.items():
            fig = plot_subplot(fig, subplot, data, opts["grid"], True)
            fig.savefig(f"plots/{name}.{filetype}")
            plt.close(fig)

    else:
        for org in orgs:
            fig = plt.figure(
                dpi=opts["dpi"], figsize=tuple([e / 2.54 for e in opts["figsize"]])
            )
            for _, subplot in subplots.items():
                fig = plot_subplot(fig, subplot, data[org], opts["grid"], False)
            fig.savefig(f"{outputFolder}/{name}_{org}.{filetype}")
            plt.close(fig)


def limit_data(data: ndarray, limits: List[int]) -> ndarray:
    return data[limits[0] : limits[1]]


def get_data_list(key: str, conf: Dict, opts_: Dict, data: Dict | None = None) -> Dict:
    if data is None:
        data = {}
    d = get_data(conf["fname"])
    dic = {}
    if "limits" in opts_ and opts_["limits"] != [None, None]:
        d = limit_data(d, opts_["limits"])
    for k, opts in opts_["data"].items():
        if bool(opts) is False:
            continue
        if "comp_offset" in opts and opts["comp_offset"]:
            d[:, opts["col"]] = (
                d[:, opts["col"]]
                - (np.max(d[:, opts["col"]]) + np.min(d[:, opts["col"]])) / 2
            )
        if "numdiff" in opts:
            da = numdiff(d, opts["col"], opts["numdiff"]) * opts["factor"]
        elif "numint" in opts:
            da = numint(d, opts["col"], opts["numdiff"]) * opts["factor"]
        else:
            da = d[:, opts["col"]] * opts["factor"]
        if "shift" in opts:
            da = da + opts["shift"]
        dic.update({k: da})

    data.update({key: dic})
    return data


def get_composite_data_list(key: str, opts_: Dict, data: Dict) -> Dict:
    dat = data[key]
    if "composite_data" not in opts_:
        return data
    for k, opts in opts_["composite_data"].items():
        d = dat[opts["base"]].copy()
        if bool(opts) is False:
            continue
        if "expr" in opts and bool(opts["expr"]) is not False:
            exec(
                "d=" + opts["expr"].replace("data[", "dat[")
            )  # format d*data["different data vector"]
        if "exp" in opts and bool(opts["exp"]) is not False:
            if isinstance(opts["exp"], str):
                d = np.exp(dat[opts["exp"]] * d)
            else:
                d = np.exp(opts["exp"] * d)
        if "frac" in opts and bool(opts["frac"]) is not False:
            if isinstance(opts["frac"], str):
                d = d / dat[opts["frac"]]
            else:
                d = d / opts["frac"]
        if "mult" in opts and bool(opts["mult"]) is not False:
            if isinstance(opts["mult"], str):
                d = d * dat[opts["mult"]]
            else:
                d = d * opts["mult"]
        if "add" in opts and bool(opts["add"]) is not False:
            if isinstance(opts["add"], str):
                d = d + dat[opts["add"]]
            else:
                d = d + opts["add"]
        if "sub" in opts and bool(opts["sub"]) is not False:
            if isinstance(opts["sub"], str):
                d = d - dat[opts["sub"]]
            else:
                d = d - opts["sub"]
        if "numdiff" in opts and bool(opts["numdiff"]) is not False:
            d = np.gradient(d, dat[opts["numdiff"]])
        if "numint" in opts and bool(opts["numint"]) is not False:
            d = np.trapz(d, dat[opts["numint"]])
        data[key].update({k: d})
    return data


def apply_plot_defaults(conf: Dict, defaults: Dict, from_conf: Dict | None) -> Dict:
    if from_conf is None:
        from_conf = {}
    subplot_default = defaults["subplots"]
    data_default = defaults["data"]
    opts = defaults.copy()
    opts.pop("subplots")
    opts.pop("data")
    uconf = deep_update(from_conf, conf)
    opts.update({"subplots": {}})
    for key in uconf["subplots"].keys():
        if not re.search(r"\d+,\d+", key):
            continue
        opts["subplots"].update({key: subplot_default})
    opts.update({"data": {}})
    for key in uconf["data"].keys():
        opts["data"].update({key: data_default})
    opts = deep_update(opts, uconf)
    return opts


def analyse_data(data: Dict) -> None:
    dic: Dict = {}
    for name, meas in data.items():
        d = {}
        for n, datap in meas.items():
            dp = {
                "max": np.max(datap),
                "min": np.min(datap),
                "avg": np.average(datap),
                "mean": np.mean(datap),
                "median": np.median(datap),
            }
            for m, ddata in meas.items():
                if n == m:
                    continue
                diff = np.diff(datap) / np.diff(ddata)
                dp.update(
                    {
                        f"max d/d{m}": np.max(diff),
                        f"min d/d{m}": np.min(diff),
                        f"avg d/d{m}": np.average(diff),
                        f"mean d/d{m}": np.mean(diff),
                        f"median d/d{m}": np.median(diff),
                    }
                )
            d.update({n: dp})
        dic.update({name: d})
    with open("./analysis.json", "w") as f:
        json.dump(dic, f, indent=2)
    return None


def main():
    global inputFolder
    global outputFolder
    global filetype
    with open("plot_config.json", "r") as f:
        config = json.load(f)
    data_defaults = config["data"]["default"]
    config["data"].pop("default")
    inputFolder = config["data"].pop("inputFolder")
    outputFolder = config["plots"].pop("outputFolder")
    filetype = config["plots"].pop("filetype")
    data = {}
    for name, conf in config["data"].items():
        opts = data_defaults
        opts = deep_update(opts, conf["opts"])
        data = get_data_list(name, conf, opts, data)

    for name, conf in config["data"].items():
        opts = data_defaults
        opts = deep_update(opts, conf["opts"])
        data = get_composite_data_list(name, opts, data)
    analyse_data(data)

    plot_defaults = config["plots"]["default"]
    config["plots"].pop("default")
    for name, conf in config["plots"].items():
        from_p = None
        if "from" in conf and conf["from"] is not None:
            from_p = config["plots"][conf["from"]]
        opts = apply_plot_defaults(conf, plot_defaults, from_p)
        do_stuff(name, opts, data)


if __name__ == "__main__":
    main()
