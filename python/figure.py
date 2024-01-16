import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import oed
import load


def trajectory_old(r):
    alpha, x_peak, y_peak, dt, T, f, v, time, tau = oed.constant("setting")

    fig1 = plt.figure(figsize=(12, 6))
    ax1 = fig1.add_subplot(111)

    # Starting Point と Gradient Peak の座標を設定
    starting_point = [0, 0]
    x_peak, y_peak = [x_peak, y_peak]

    # Starting Point と Gradient Peak をプロット
    ax1.scatter(*starting_point, color="black", label="Starting Point")
    ax1.scatter(x_peak, y_peak, color="red", label="Gradient Peak")

    segments = []
    for n in np.arange(len(r[0]) - 1):
        segments.append(np.array([[r[0, n], r[1, n]], [r[0, n + 1], r[1, n + 1]]]))

    lc = LineCollection(segments, cmap="jet", linewidth=1.5)
    cols = np.linspace(0, time, len(segments))
    lc.set_array(cols)

    ax1.add_collection(lc)

    # x軸とy軸の範囲を設定
    ax1.set_xlim(np.min(r[0]) - 0.1, np.max(r[0]) + 0.5)
    ax1.set_ylim(np.min(r[1]) - 0.1, np.max(r[1]) + 0.1)

    # カラーバーを追加し、位置をフィギュア内の右下に指定
    cax = fig1.add_axes([0.83, 0.15, 0.02, 0.4])  # [left, bottom, width, height]
    colorbar = plt.colorbar(lc, cax=cax)  # shrink パラメータでサイズを調整
    colorbar.set_label("Time")  # カラーバーのラベルを設定

    # 凡例を表示
    ax1.legend()

    plt.show()

    return


def single_line_stacks(x, y):
    lines = []
    for coord in range(1, len(x), 2):
        x_elems = x[coord - 1 : coord + 2]
        y_elems = y[coord - 1 : coord + 2]
        lines.append(np.column_stack([x_elems, y_elems]))
    return lines


def trajectory(gene, lines_number, out_file_path):
    fig, ax = plt.subplots(figsize=(10, 7))

    # トラジェクトリーの表示
    for idx, angle in enumerate(np.linspace(0, 2 * np.pi, lines_number + 1)):
        r = oed.klinotaxis(gene, angle)
        alpha, x_peak, y_peak, dt, T, f, v, time, tau = oed.constant("setting")

        lines = single_line_stacks(r[0], r[1])
        color = np.linspace(0, time, len(lines))
        lc = LineCollection(lines, cmap="jet", linewidth=1, array=color)
        line = ax.add_collection(lc)

        if idx == int(lines_number / 2 + 1):
            lc_inset = LineCollection(lines, cmap="jet", linewidth=1, array=color)
            ins_x_min = min(r[0])
            ins_y_min = min(r[1])

    # スタートとゴールの表示
    starting_point = [0, 0]
    peak = [x_peak, y_peak]

    ax.scatter(*starting_point, s=15, color="black")
    ax.scatter(*peak, s=15, color="black")

    y_max = 1
    ax.vlines(
        starting_point[0],
        starting_point[1],
        y_max,
        color="black",
        linestyle="-",
        linewidth=0.5,
    )
    ax.vlines(peak[0], peak[1], y_max, color="black", linestyle="-", linewidth=0.5)

    ax.text(
        starting_point[0], y_max + 0.1, "Starting Point", horizontalalignment="center"
    )
    ax.text(peak[0], y_max + 0.1, "Gradient Peak", horizontalalignment="center")

    # 軸メモリや枠を非表示にする
    ax.axis("off")
    ax.autoscale()
    ax.set_aspect("equal")

    # 基準の大きさを表示
    ax.text(4.5, -0.95, "1 cm", horizontalalignment="center")
    ax.hlines(-1, 4, 5, color="black", linestyle="-", linewidth=1.5)

    # カラーバーの縦の大きさを変更
    plt.colorbar(line, ax=ax, label="time /s", shrink=0.5)

    # インセットプロット
    axins = ax.inset_axes([-0.1, -0.7, 0.8, 0.8])
    axins.add_collection(lc_inset)

    axins.set_xlim(ins_x_min - 0.1, ins_x_min + 1.4)
    axins.set_ylim(ins_y_min - 0.1, ins_y_min + 0.9)

    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_edgecolor("gray")

    axins.text(ins_x_min + 1.05, ins_y_min + 0.25, "1 mm", horizontalalignment="center")
    axins.hlines(
        ins_y_min + 0.2,
        ins_x_min + 1,
        ins_x_min + 1.1,
        color="black",
        linestyle="-",
        linewidth=1.5,
    )

    ax.indicate_inset_zoom(axins)

    # グラフの保存および表示
    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return


def newron_output(gene):
    N, M, theta, w_on, w_off, w, g, w_osc, w_nmj = oed.weight(gene)
    alpha, x_peak, y_peak, dt, T, f, v, time, tau = oed.constant(
        "setting_newron_output"
    )
    start = 150  # 開始時間

    # ラベル
    name = [
        "ASEL",
        "ASER",
        "AIYL",
        "AIYR",
        "AIZL",
        "AIZR",
        "SMBDL",
        "SMBDR",
        "SMBVL",
        "SMBVR",
    ]
    plot_index = [3, 4, 5, 6, 9, 7, 8, 10]

    # 各種配列の初期化
    t = np.arange(0, time, dt)
    y = np.zeros((8, len(t)))

    # figsize
    plt.figure(figsize=(8, 8))

    def ASE_line(ASE_mode):
        for newron in np.arange(0, 1, 0.1):
            ASEL = np.zeros(len(t))
            ASER = np.zeros(len(t))

            if ASE_mode == 0:
                for i in range(int(2 / dt)):
                    ASEL[int(4 / dt) + i] = newron * (-t[i] / 2 - np.floor(-t[i] / 2))
            else:
                for i in range(int(2 / dt)):
                    ASER[int(4 / dt) + i] = newron * (-t[i] / 2 - np.floor(-t[i] / 2))

            # オイラー法
            for k in range(len(t) - 1):
                # シナプス結合およびギャップ結合からの入力
                synapse = np.dot(w.T, oed.sigmoid(y[:, k] + theta))
                gap = np.array([np.dot(g[:, i], (y[:, k] - y[i, k])) for i in range(8)])

                # 介在ニューロンおよび運動ニューロンの膜電位の更新
                y[:, k + 1] = (
                    y[:, k]
                    + (
                        -y[:, k]
                        + synapse
                        + gap
                        + w_on * ASEL[k]
                        + w_off * ASER[k]
                        + w_osc * oed.y_osc(t[k], T)
                    )
                    / tau
                    * dt
                )

            # カラーマップを使用して色を指定
            if ASE_mode == 0:
                color = plt.cm.Blues(newron)
            else:
                color = plt.cm.Reds(newron)

            # プロットを指定した位置に表示
            plt.subplot(5, 2, 1)
            plt.plot(t[start:], ASEL[start:], color=color)
            plt.subplot(5, 2, 2)
            plt.plot(t[start:], ASER[start:], color=color)
            for i in range(8):
                plt.subplot(5, 2, plot_index[i])
                plt.plot(t[start:], oed.sigmoid(y[i] + theta[i])[start:], color=color)

        return

    ASE_line(0)
    ASE_line(1)

    def black_line():
        ASEL = np.zeros(len(t))
        ASER = np.zeros(len(t))
        # オイラー法
        for k in range(len(t) - 1):
            # シナプス結合およびギャップ結合からの入力
            synapse = np.dot(w.T, oed.sigmoid(y[:, k] + theta))
            gap = np.array([np.dot(g[:, i], (y[:, k] - y[i, k])) for i in range(8)])

            # 介在ニューロンおよび運動ニューロンの膜電位の更新
            y[:, k + 1] = (
                y[:, k]
                + (
                    -y[:, k]
                    + synapse
                    + gap
                    + w_on * ASEL[k]
                    + w_off * ASER[k]
                    + w_osc * oed.y_osc(t[k], T)
                )
                / tau
                * dt
            )

        # プロットを指定した位置に表示
        ax = plt.subplot(5, 2, 1)
        plt.plot(t[start:], ASEL[start:], color="black")
        plt.title(name[0])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax = plt.subplot(5, 2, 2)
        plt.plot(t[start:], ASER[start:], color="black")
        plt.title(name[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        for i in range(8):
            ax = plt.subplot(5, 2, plot_index[i])
            plt.plot(t[start:], oed.sigmoid(y[i] + theta[i])[start:], color="black")
            plt.title(name[plot_index[i] - 1])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
        plt.show()

        return

    black_line()

    return


def Bearing_vs_Turing_bias(in_file_path, out_file_path):
    data = load.load_output_txt(in_file_path)

    plt.errorbar(
        data[0],
        data[1],
        yerr=data[2],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="black",
        markeredgecolor="black",
        color="black",
    )

    plt.xlabel("Bearing (degrees)")
    plt.ylabel("Turning bias (degrees)")
    plt.xlim(-185, 185)
    plt.xticks([-180, -90, 0, 90, 180])
    plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()


def Normal_gradient_vs_Turing_bias(in_file_path, out_file_path):
    data = load.load_output_txt(in_file_path)

    plt.errorbar(
        data[0],
        data[1],
        yerr=data[2],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="black",
        markeredgecolor="black",
        color="black",
    )

    plt.xlabel("Normal_gradient (mM/cm)")
    plt.ylabel("Turning bias (degrees)")
    plt.xticks([-0.01, -0.005, 0, 0.005, 0.01])
    plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()


def Translational_gradient_vs_Turing_bias(in_file_path, out_file_path):
    data = load.load_output_txt(in_file_path)

    plt.scatter(data[0], data[1], s=6, c="black", marker="o", edgecolors="black")

    plt.errorbar(
        data[0],
        data[3],
        yerr=data[4],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="red",
        markeredgecolor="red",
        color="red",
    )

    plt.errorbar(
        data[0],
        data[5],
        yerr=data[6],
        capsize=5,
        fmt="o",
        markersize=3,
        ecolor="blue",
        markeredgecolor="blue",
        color="blue",
    )

    plt.xlabel("Normal_gradient (mM/cm)")
    plt.ylabel("Turning bias (degrees)")
    plt.xticks([-0.01, -0.005, 0, 0.005, 0.01])
    plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()
