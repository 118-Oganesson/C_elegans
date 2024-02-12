import load
import os
import json
import numpy as np


def gene_aser_synapse_change(gene_number, start, stop, num):
    """
    gene_number: 改変する遺伝子の番号
    start: 10,11の結合に対して加える範囲の最小
    stop: 10,11の結合に対して加える範囲の最大
    num: 範囲の分割数
    """

    result = load.load_result_json("../result/Result_aiz_negative.json")
    base_gene = np.array(result[gene_number]["gene"])

    change_result = []
    for delta_gene in np.linspace(start, stop, num + 1):
        zero_gene = np.zeros_like(base_gene)
        zero_gene[10:12] = [delta_gene] * 2
        change_gene = base_gene + zero_gene
        change_result.append({"value": delta_gene, "gene": change_gene.tolist()})

    json_data = json.dumps(change_result, indent=1)
    os.makedirs("../result/concentration_memory", exist_ok=True)
    with open(
        "../result/concentration_memory/Result_aiz_negative_{}.json".format(
            gene_number
        ),
        "w",
    ) as json_file:
        json_file.write(json_data)
