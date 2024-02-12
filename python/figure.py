import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import oed
import load


def trajectory_old(r):
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = oed.constant(
        "setting"
    )

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


def calculate_trajectory(gene_angle_list):
    gene, angle, c_mode = gene_angle_list
    return oed.klinotaxis(gene, angle, c_mode)


def trajectory(gene, c_mode, lines_number, zoom_number, out_file_path):
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = oed.constant(
        "setting"
    )
    fig, ax = plt.subplots(figsize=(10, 7))

    # マルチスレッドの使用プロセス数
    if lines_number < multiprocessing.cpu_count():
        process = lines_number
    else:
        process = multiprocessing.cpu_count()

    # マルチスレッドで処理する遺伝子と角度のリスト
    gene_angle_list = [
        [gene, angle, c_mode]
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / lines_number)
    ]

    # マルチスレッド処理
    with multiprocessing.Pool(process) as pool:
        results = pool.map(calculate_trajectory, gene_angle_list)

    # トラジェクトリーの表示
    for idx, r in enumerate(results):
        lines = single_line_stacks(r[0], r[1])
        color = np.linspace(0, time, len(lines))
        lc = LineCollection(lines, cmap="jet", linewidth=1, array=color)
        line = ax.add_collection(lc)

        if idx == zoom_number:
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


def newron_output(gene, out_file_path):
    N, M, theta, w_on, w_off, w, g, w_osc, w_nmj = oed.weight(gene)
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = oed.constant(
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

        return

    black_line()

    plt.savefig(out_file_path, dpi=300)
    plt.show()

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
    plt.ylabel("Turning bias (degrees/cm)")
    plt.xlim(-185, 185)
    plt.xticks([-180, -90, 0, 90, 180])
    # plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return


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

    plt.xlabel("Normal gradient (mM/cm)")
    plt.ylabel("Turning bias (degrees/cm)")
    # plt.xticks([-0.01, -0.005, 0, 0.005, 0.01])
    # plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return


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

    plt.xlabel("Translational gradient (mM/cm)")
    plt.ylabel("Turning bias (degrees/cm)")
    # plt.xticks([-0.01, -0.005, 0, 0.005, 0.01])
    # plt.yticks([-40, -20, 0, 20, 40])

    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return


def angle_between_vectors(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    cos_theta = dot_product / (norm_a * norm_b)
    radians = np.arccos(cos_theta)
    degrees = np.degrees(radians)

    return degrees


def normal_line(point_1, point_2, width):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    vector = point_2 - point_1
    magnitude_vector = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    normal = np.array([-vector[1], vector[0]]) / magnitude_vector
    normal_rev = -normal

    point_3 = point_2 + normal * width / 2
    point_4 = point_2 + normal_rev * width / 2

    return [point_3[0], point_4[0]], [point_3[1], point_4[1]]


def annotation_head_short(point_1, point_2, head_length):
    point_1 = np.array(point_1)
    point_2 = np.array(point_2)
    vector = point_1 - point_2
    magnitude_vector = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    short_vector = vector / magnitude_vector * head_length

    return point_2 + short_vector


def connectome(gene, out_file_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    # 結合の設定
    headwidth = 9
    headlength = 10
    width = 4
    gap_width = 6
    inv_root_2 = 1 / np.sqrt(2)

    positive_color = "blue"
    negative_color = "red"
    gap_color = "green"

    chemosensory_newrons = [(3, 9), (6, 9)]
    inter_newrons = [(3, 6), (6, 6), (3, 3), (6, 3)]
    motor_newrons = [(0, 0), (3, 0), (6, 0), (9, 0)]

    chemosensory_names = ["ASEL", "ASER"]
    inter_names = ["AIYL", "AIYR", "AIZL", "AIZR"]
    motor_names = ["SMBVL", "SMBDL", "SMBDR", "SMBVR"]

    # 円の作成
    # 感覚ニューロン
    for i, center in enumerate(chemosensory_newrons):
        circle = patches.Circle(
            xy=center, radius=1, facecolor="white", edgecolor="black"
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1],
            chemosensory_names[i],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # 介在ニューロン
    for i, center in enumerate(inter_newrons):
        circle = patches.Circle(
            xy=center, radius=1, facecolor="lightgray", edgecolor="black"
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1],
            inter_names[i],
            color="black",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # 運動ニューロン
    for i, center in enumerate(motor_newrons):
        circle = patches.Circle(
            xy=center, radius=1, facecolor="black", edgecolor="black"
        )
        ax.add_patch(circle)
        ax.text(
            center[0],
            center[1],
            motor_names[i],
            color="white",
            horizontalalignment="center",
            verticalalignment="center",
        )

    # 結合の作成

    # 感覚ニューロン
    chemosensory_annotations = [
        {
            "start": (chemosensory_newrons[0][0], chemosensory_newrons[0][1] - 1),
            "end": (chemosensory_newrons[0][0], chemosensory_newrons[0][1] - 2),
            "weight": gene[8],
        },
        {
            "start": (chemosensory_newrons[1][0], chemosensory_newrons[1][1] - 1),
            "end": (chemosensory_newrons[1][0], chemosensory_newrons[1][1] - 2),
            "weight": gene[11],
        },
        {
            "start": (
                chemosensory_newrons[0][0] + inv_root_2,
                chemosensory_newrons[0][1] - inv_root_2,
            ),
            "end": (inter_newrons[1][0] - inv_root_2, inter_newrons[1][1] + inv_root_2),
            "weight": gene[9],
        },
        {
            "start": (
                chemosensory_newrons[1][0] - inv_root_2,
                chemosensory_newrons[1][1] - inv_root_2,
            ),
            "end": (inter_newrons[0][0] + inv_root_2, inter_newrons[0][1] + inv_root_2),
            "weight": gene[10],
        },
    ]

    for annotation in chemosensory_annotations:
        if annotation["weight"] > 0:
            color = positive_color
            arrow = dict(
                shrink=0,
                width=np.abs(annotation["weight"]) * width,
                headwidth=headwidth,
                headlength=headlength,
                connectionstyle="arc3",
                facecolor=color,
                edgecolor=color,
            )
        else:
            color = negative_color
            arrow = dict(
                arrowstyle="|-|, widthA=0, widthB=0.4",
                linewidth=np.abs(annotation["weight"]) * width,
                facecolor=color,
                edgecolor=color,
            )
        ax.annotate(
            "", xy=annotation["end"], xytext=annotation["start"], arrowprops=arrow
        )

    # 介在ニューロン
    inter_annotations = [
        {
            "start": (inter_newrons[0][0], inter_newrons[0][1] - 1),
            "end": (inter_newrons[0][0], inter_newrons[0][1] - 2),
            "weight": gene[12],
        },
        {
            "start": (inter_newrons[1][0], inter_newrons[1][1] - 1),
            "end": (inter_newrons[1][0], inter_newrons[1][1] - 2),
            "weight": gene[13],
        },
        {
            "start": (inter_newrons[2][0], inter_newrons[2][1] - 1),
            "end": (inter_newrons[2][0], inter_newrons[2][1] - 2),
            "weight": gene[14],
        },
        {
            "start": (inter_newrons[3][0], inter_newrons[3][1] - 1),
            "end": (inter_newrons[3][0], inter_newrons[3][1] - 2),
            "weight": gene[15],
        },
        {
            "start": (
                inter_newrons[2][0] - inv_root_2,
                inter_newrons[2][1] - inv_root_2,
            ),
            "end": (motor_newrons[0][0] + inv_root_2, motor_newrons[0][1] + inv_root_2),
            "weight": gene[14],
        },
        {
            "start": (
                inter_newrons[3][0] + inv_root_2,
                inter_newrons[3][1] - inv_root_2,
            ),
            "end": (motor_newrons[3][0] - inv_root_2, motor_newrons[3][1] + inv_root_2),
            "weight": gene[15],
        },
    ]

    for annotation in inter_annotations:
        if annotation["weight"] > 0:
            color = positive_color
            arrow = dict(
                shrink=0,
                width=np.abs(annotation["weight"]) * width,
                headwidth=headwidth,
                headlength=headlength,
                connectionstyle="arc3",
                facecolor=color,
                edgecolor=color,
            )
        else:
            color = negative_color
            arrow = dict(
                arrowstyle="|-|, widthA=0, widthB=0.4",
                linewidth=np.abs(annotation["weight"]) * width,
                facecolor=color,
                edgecolor=color,
            )
        ax.annotate(
            "", xy=annotation["end"], xytext=annotation["start"], arrowprops=arrow
        )

    # 運動ニューロン
    base = np.array([1 / 2, 0])
    start = np.array(
        [(np.sqrt(2) + np.sqrt(30)) / 16, (-1 / 8 + np.sqrt(15) / 8) / np.sqrt(2)]
    )
    end = np.array(
        [(np.sqrt(2) - np.sqrt(30)) / 16, (-1 / 8 - np.sqrt(15) / 8) / np.sqrt(2)]
    )

    start_angle = angle_between_vectors(base, start)
    end_angle = 360 - angle_between_vectors(base, end)

    motor_curves = [
        {
            "center": (
                motor_newrons[0][0] - inv_root_2,
                motor_newrons[0][1] + inv_root_2,
            ),
            "weight": gene[16],
            "start_angle": start_angle,
            "end_angle": end_angle,
            "point_1": (motor_newrons[0][0], motor_newrons[0][1]),
            "point_2": (
                motor_newrons[0][0] - inv_root_2 + end[0],
                motor_newrons[0][1] + inv_root_2 + end[1],
            ),
        },
        {
            "center": (
                motor_newrons[1][0] - inv_root_2,
                motor_newrons[1][1] + inv_root_2,
            ),
            "weight": gene[16],
            "start_angle": start_angle,
            "end_angle": end_angle,
            "point_1": (motor_newrons[1][0], motor_newrons[1][1]),
            "point_2": (
                motor_newrons[1][0] - inv_root_2 + end[0],
                motor_newrons[1][1] + inv_root_2 + end[1],
            ),
        },
        {
            "center": (
                motor_newrons[2][0] + inv_root_2,
                motor_newrons[2][1] + inv_root_2,
            ),
            "weight": gene[17],
            "start_angle": -(end_angle - 180),
            "end_angle": 180 - start_angle,
            "point_1": (motor_newrons[2][0], motor_newrons[2][1]),
            "point_2": (
                motor_newrons[2][0] - inv_root_2 + 2 * inv_root_2 - end[0],
                motor_newrons[2][1] + inv_root_2 + end[1],
            ),
        },
        {
            "center": (
                motor_newrons[3][0] + inv_root_2,
                motor_newrons[3][1] + inv_root_2,
            ),
            "weight": gene[17],
            "start_angle": -(end_angle - 180),
            "end_angle": 180 - start_angle,
            "point_1": (motor_newrons[3][0], motor_newrons[3][1]),
            "point_2": (
                motor_newrons[3][0] - inv_root_2 + 2 * inv_root_2 - end[0],
                motor_newrons[3][1] + inv_root_2 + end[1],
            ),
        },
    ]

    for curve in motor_curves:
        if curve["weight"] > 0:
            color = positive_color
            ax.annotate(
                "",
                xy=annotation_head_short(curve["point_1"], curve["point_2"], 0.25),
                xytext=curve["point_2"],
                arrowprops=dict(
                    headwidth=headwidth,
                    headlength=headlength,
                    connectionstyle="arc3",
                    facecolor=color,
                    edgecolor=color,
                ),
            )
        else:
            color = negative_color
            point = normal_line(
                curve["point_1"],
                curve["point_2"],
                0.3,
            )
            ax.plot(
                point[0],
                point[1],
                color=color,
                linewidth=np.abs(curve["weight"]) * width,
            )

        self = patches.Arc(
            xy=curve["center"],
            width=1,
            height=1,
            theta1=curve["start_angle"],
            theta2=curve["end_angle"],
            facecolor=color,
            edgecolor=color,
            linewidth=np.abs(curve["weight"]) * width,
        )
        ax.add_patch(self)

    # ギャップ結合の作成
    inter_lines = [
        {
            "base": inter_newrons[0][1],
            "start": inter_newrons[0][0] + 1,
            "end": inter_newrons[1][0] - 1,
            "weight": oed.gene_range_1(gene[18], 0, gap_width),
        },
        {
            "base": inter_newrons[2][1],
            "start": inter_newrons[2][0] + 1,
            "end": inter_newrons[3][0] - 1,
            "weight": oed.gene_range_1(gene[19], 0, gap_width),
        },
    ]

    for line in inter_lines:
        ax.hlines(
            line["base"],
            line["start"],
            line["end"],
            color=gap_color,
            linestyle="-",
            linewidth=line["weight"],
        )

    ax.axis("off")
    ax.autoscale()
    ax.set_aspect("equal")

    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return


def calculate_trajectory_membrane_potential(gene_angle_list):
    gene, angle, c_mode = gene_angle_list

    r, y = oed.klinotaxis_membrane_potential(gene, angle, c_mode)

    lines = single_line_stacks(r[0], r[1])
    aiy = (y[0] + y[1]) / 2
    aiz = (y[2] + y[3]) / 2

    return lines, aiy, aiz


def trajectory_membrane_potential(gene, c_mode, lines_number, out_file_path):
    alpha, x_peak, y_peak, dt, T, f, v, time, tau, c_0, lambda_ = oed.constant(
        "setting"
    )
    fig, ax = plt.subplots(2, 1, figsize=(10, 7))

    top = 0
    bottom = 1
    cmap = "seismic"

    # マルチスレッドの使用プロセス数
    if lines_number < multiprocessing.cpu_count():
        process = lines_number
    else:
        process = multiprocessing.cpu_count()

    # マルチスレッドで処理する遺伝子と角度のリスト
    gene_angle_c_mode_list = [
        [gene, angle, c_mode]
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / lines_number)
    ]

    # マルチスレッド処理
    with multiprocessing.Pool(process) as pool:
        results = pool.map(
            calculate_trajectory_membrane_potential, gene_angle_c_mode_list
        )

    # 結果を分解して格納
    all_lines, all_aiy, all_aiz = zip(*results)

    flat_aiy = np.array(all_aiy).flatten()
    flat_aiz = np.array(all_aiz).flatten()

    mean_aiy = np.mean(flat_aiy)
    std_dev_aiy = np.std(flat_aiy)
    mean_aiz = np.mean(flat_aiz)
    std_dev_aiz = np.std(flat_aiz)

    # AIYの表示
    for i, lines in enumerate(all_lines):
        lc = LineCollection(
            lines,
            cmap=cmap,
            norm=Normalize(vmin=mean_aiy - std_dev_aiy, vmax=mean_aiy + std_dev_aiy),
            linewidth=1,
            array=all_aiy[i],
        )
        line_aiy = ax[top].add_collection(lc)

    # AIZの表示
    for i, lines in enumerate(all_lines):
        lc = LineCollection(
            lines,
            cmap=cmap,
            norm=Normalize(vmin=mean_aiz - std_dev_aiz, vmax=mean_aiz + std_dev_aiz),
            linewidth=1,
            array=all_aiz[i],
        )
        line_aiz = ax[bottom].add_collection(lc)

    # スタートとゴールの表示
    starting_point = [0, 0]
    peak = [x_peak, y_peak]

    for i in [top, bottom]:
        ax[i].scatter(*starting_point, s=15, color="black")
        ax[i].scatter(*peak, s=15, color="black")

        y_max = 1
        ax[i].vlines(
            starting_point[0],
            starting_point[1],
            y_max,
            color="black",
            linestyle="-",
            linewidth=0.5,
        )
        ax[i].vlines(
            peak[0], peak[1], y_max, color="black", linestyle="-", linewidth=0.5
        )

        ax[i].text(
            starting_point[0],
            y_max + 0.1,
            "Starting Point",
            horizontalalignment="center",
        )
        ax[i].text(peak[0], y_max + 0.1, "Gradient Peak", horizontalalignment="center")

    # 軸メモリや枠を非表示にする
    for i in [top, bottom]:
        ax[i].axis("off")
        ax[i].autoscale()
        ax[i].set_aspect("equal")

    # 基準の大きさを表示
    for i in [top, bottom]:
        ax[i].text(4.5, -0.95, "1 cm", horizontalalignment="center")
        ax[i].hlines(-1, 4, 5, color="black", linestyle="-", linewidth=1.5)

    # カラーバー表示
    plt.colorbar(line_aiy, ax=ax[top], label="AIY membrane potential", shrink=0.5)
    plt.colorbar(line_aiz, ax=ax[bottom], label="AIZ membrane potential", shrink=0.5)

    # タイトルの表示
    ax[top].set_title("AIY")
    ax[bottom].set_title("AIZ")

    # グラフの保存および表示
    plt.savefig(out_file_path, dpi=300)
    plt.show()

    return
