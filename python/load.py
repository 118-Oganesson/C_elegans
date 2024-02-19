import toml
import json


def load_simulation_setting_toml(file_path):
    """
    file_path: 読み込むシミュレーション設定に関するtomlファイルのパス
    """
    with open(file_path, "r") as toml_file:
        data = toml.load(toml_file)
    return data


def load_result_json(file_path):
    """
    file_path: 読み込む遺伝子に関するjsonファイルのパス
    """
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
    return data


def load_output_txt(file_path):
    """
    file_path: 読み込むアウトプットに関するtxtファイルのパス

    データは列で読み込む（文字の分割は", "で行っている）
    """
    with open(file_path, "r") as txt_file:
        # ファイルの各行を読み込んでリストに格納
        lines = txt_file.read().split("\n")

    # 各行での最大列数を求める
    max_columns = max(len(line.split(", ")) for line in lines if line)

    # 列ごとのデータを格納するためのリストを初期化
    columns = [[] for _ in range(max_columns)]

    # 行ごとにデータを列に振り分ける
    for line in lines:
        if line:
            # 各行のデータをカンマで分割し、文字列からfloat64に変換
            values = [float(value) for value in line.split(", ")]
            # 各列にデータを振り分け
            for i in range(len(values)):
                columns[i].append(values[i])

    return columns
