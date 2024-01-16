import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import oed


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


def trajectory(gene, lines_number):
    fig, ax = plt.subplots()

    for angle in np.linspace(0, 2 * np.pi, lines_number + 1):
        r = oed.klinotaxis(gene, angle)
        alpha, x_peak, y_peak, dt, T, f, v, time, tau = oed.constant("setting")

        lines = single_line_stacks(r[0], r[1])
        color = np.linspace(0, time, len(lines))
        lc = LineCollection(lines, cmap="jet", linewidth=1.5, array=color)
        line = ax.add_collection(lc)

    starting_point = [0, 0]
    x_peak, y_peak = [x_peak, y_peak]

    ax.scatter(*starting_point, color="black", label="Starting Point")
    ax.scatter(x_peak, y_peak, color="red", label="Gradient Peak")

    # 軸メモリや枠を非表示にする
    ax.axis("off")
    ax.autoscale()
    ax.set_aspect("equal")

    # 基準の大きさを表示
    ax.text(4.5, -0.9, "1 cm")
    ax.hlines(-1, 4, 5, color="black", linestyle="-", linewidth=1.5)

    # カラーバーの縦の大きさを変更
    plt.colorbar(line, ax=ax, label="Time", shrink=0.5)
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
