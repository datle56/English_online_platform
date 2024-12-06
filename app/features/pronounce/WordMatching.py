import numpy as np
from scipy.optimize import linear_sum_assignment

# Hàm tính toán khoảng cách Levenshtein (Dynamic Programming)
def compute_levenshtein_distance(str1, str2):
    m = len(str1)
    n = len(str2)
    dp = np.zeros((m + 1, n + 1), dtype=int)

    # Khởi tạo ma trận DP
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Tính toán khoảng cách Levenshtein
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    # In ra ma trận DP (tuỳ chọn)
    # print("Ma trận DP (Levenshtein Distance Matrix):")
    # print(dp)

    return dp[m][n]

# Hàm tính toán ma trận khoảng cách giữa hai danh sách từ
def compute_word_distance_matrix(words1, words2):
    distance_matrix = np.zeros((len(words1), len(words2)), dtype=int)

    for i in range(len(words1)):
        for j in range(len(words2)):
            distance_matrix[i][j] = compute_levenshtein_distance(words1[i], words2[j])

    return distance_matrix

# Hàm sắp xếp lại predicted_words với ký tự placeholder
def align_predicted_to_real_with_placeholder(real_words, predicted_words, placeholder="#"):
    # Tách câu thành danh sách các từ
    real_words_list = real_words.split()
    predicted_words_list = predicted_words.split()

    # Tính toán ma trận khoảng cách giữa các từ
    distance_matrix = compute_word_distance_matrix(real_words_list, predicted_words_list)

    # Sử dụng thuật toán Hungarian để tìm ghép tối ưu
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Tạo một danh sách để lưu các từ đã ghép
    alignment = [placeholder] * len(real_words_list)

    # Tạo một tập để theo dõi các từ đã được ghép trong predicted_words
    used_predicted = set()

    for i, j in zip(row_ind, col_ind):
        # Bạn có thể đặt một ngưỡng để quyết định xem có ghép hay không
        # Ví dụ: chỉ ghép nếu khoảng cách Levenshtein nhỏ hơn hoặc bằng 2
        if distance_matrix[i][j] <= 2:
            alignment[i] = predicted_words_list[j]
            used_predicted.add(j)

    # Nếu cần, bạn có thể xử lý các từ dự đoán thừa ở đây

    return ' '.join(alignment)