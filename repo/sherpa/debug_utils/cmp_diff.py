import numpy as np
import sys
import os


import numpy as np

# 计算两个矩阵的余弦相似度
def cosine_similarity(matrix1, matrix2):
    # 确保两个矩阵形状相同
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    # 计算余弦相似度
    dot_product = np.dot(matrix1.flatten(), matrix2.flatten())
    norm_matrix1 = np.linalg.norm(matrix1)
    norm_matrix2 = np.linalg.norm(matrix2)
    similarity = dot_product / (norm_matrix1 * norm_matrix2)
    
    return similarity

# 计算两个矩阵差的绝对值的平均值
def mean_absolute_difference(matrix1, matrix2):
    # 确保两个矩阵形状相同
    assert matrix1.shape == matrix2.shape, "Matrices must have the same shape."

    # 计算差的绝对值的平均值
    absolute_difference = np.abs(matrix1 - matrix2)
    mean_absolute_diff = np.mean(absolute_difference)
    
    return mean_absolute_diff

# folder1 = "/data/Kaldi-TPU/build/debug" # sys.argv[1]
# folder2 = "/data/sherpa-onnx/build-aarch64-linux-gnu/debug" # sys.argv[2]

# for file in os.listdir(folder1):
#     if file.startswith("output"):
#         continue
#     if file != "input0.npy":
#         continue
#     if ("gt_"+file) in os.listdir(folder2):
#         f1 = folder1 + "/" + file
#         f2 = folder2 + "/gt_" + file
#         x = np.load(f1)
#         gt = np.load(f2)
#         names = x.files
#         print("========================", file)
#         print('Input name  \tMAD \tCOS')
#         for key in names:
#             if key in gt.files:
#                 breakpoint()
#                 # print('{:s}  \t{:.4f} \t{:.4f}'.format(
#                 #     key,
#                 #     mean_absolute_difference(x[key], gt[key]),
#                 #     cosine_similarity(x[key], gt[key])
#                 #     )
#                 # )
#             else:
#                 print("No gt file found for key: ", key)

tpu_in_prefix = "/data/Kaldi-TPU/build/debug/tpu_input"
ort_in_prefix = "/data/Kaldi-TPU/build/debug/ort_input"
gt_in_prefix = "/data/Kaldi-TPU/build/debug/gt_input"

tpu_out_prefix = "/data/Kaldi-TPU/build/debug/tpu_output"
ort_out_prefix = "/data/Kaldi-TPU/build/debug/ort_output"
gt_out_prefix = "/data/Kaldi-TPU/build/debug/gt_output"

for i in range(10):
    print(f'==================== Forward {i} ======================')
    tpu_in = np.load(f"{tpu_in_prefix}{i}.npz")
    ort_in = np.load(f"{ort_in_prefix}{i}.npz")
    gt_in = np.load(f"{gt_in_prefix}{i}.npz")
    tpu_out = np.load(f"{tpu_out_prefix}{i}.npz")
    ort_out = np.load(f"{ort_out_prefix}{i}.npz")
    gt_out = np.load(f"{gt_out_prefix}{i}.npz")
    for idx in range(99):
        if i % 2 == 1: breakpoint()
        print(idx, mean_absolute_difference(tpu_in[f'input_{idx}'], gt_in[f'input_{idx}']), " | ", mean_absolute_difference(ort_in[f'input_{idx}'], gt_in[f'input_{idx}']))
        # print(idx, cosine_similarity(tpu_in[f'input_{idx}'], gt_in[f'input_{idx}']), " | ", cosine_similarity(ort_in[f'input_{idx}'], gt_in[f'input_{idx}']))
        print(idx, mean_absolute_difference(tpu_out[f'output_{idx}'], gt_out[f'output_{idx}']), " | ", mean_absolute_difference(ort_out[f'output_{idx}'], gt_out[f'output_{idx}']))
        # print(idx, cosine_similarity(tpu_out[f'output_{idx}'], gt_out[f'output_{idx}']), " | ", cosine_similarity(ort_out[f'output_{idx}'], gt_out[f'output_{idx}']))