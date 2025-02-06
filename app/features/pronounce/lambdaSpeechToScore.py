import torch
import json
import os
import base64
import time
import numpy as np
from transformers import Wav2Vec2Processor
from .model import CustomWav2Vec2ForCTC
from . import CharacterMatching, utilsFileIO, WordMatching
import eng_to_ipa as ipa
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 

# # Đường dẫn cho hai mô hình khác nhau
# text_model_checkpoint_dir = r"D:\DOANTOTNGHIEP\code\archive\checkpoint-text"
# ipa_model_checkpoint_dir = r"D:\DOANTOTNGHIEP\code\archive\checkpoint-ipa"

# # Load hai processor và mô hình
# text_processor = Wav2Vec2Processor.from_pretrained(text_model_checkpoint_dir)
# text_model = CustomWav2Vec2ForCTC.from_pretrained(text_model_checkpoint_dir)

# ipa_processor = Wav2Vec2Processor.from_pretrained(ipa_model_checkpoint_dir)
# ipa_model = CustomWav2Vec2ForCTC.from_pretrained(ipa_model_checkpoint_dir)

import torch
from torch import nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Encoder
from transformers import Wav2Vec2Processor

# from model import CustomWav2Vec2ForCTC  # Import mô hình từ file model.py
# from .model import CustomWav2Vec2ForCTC


# checkpoint_dir = "/content/dataset-folder/checkpoint-50000"
# processor = Wav2Vec2Processor.from_pretrained('/content/dataset-folder')
    
processor = Wav2Vec2Processor.from_pretrained(r"D:\DOANTOTNGHIEP\DOAN\be\features\pronounce\model")
model = CustomWav2Vec2ForCTC.from_pretrained(r"D:\DOANTOTNGHIEP\DOAN\be\features\pronounce\model")

# processor = Wav2Vec2Processor.from_pretrained(r"C:\Users\ngocd\Downloads\test_last_transformer_1\est_last_transformer_1")
# model = CustomWav2Vec2ForCTC.from_pretrained(r"C:\Users\ngocd\Downloads\test_last_transformer_1\est_last_transformer_1")


ipa_model = model
text_processor = processor
ipa_processor = processor
text_model = model


def compare_words(real_words, predicted_words):
    """
    So sánh hai danh sách từ (có thể là text hoặc IPA) và trả về chuỗi nhị phân khớp.
    """


    # predicted_words = WordMatching.align_predicted_to_real_with_placeholder(real_words, predicted_words)
    real_words =   real_words.lower()

    real_words = real_words.split()
    predicted_words =   predicted_words.split()



    mapped_words, mapped_words_indices = CharacterMatching.get_best_mapped_words(
        predicted_words, real_words
    )

    is_letter_correct_all_words = ''

    for idx, word_real in enumerate(real_words):
        mapped_letters, mapped_letters_indices = CharacterMatching.get_best_mapped_words(
            mapped_words[idx], word_real
        )
        print("--------------------------")
        print(mapped_letters)

        print(f"Từ thực tế: {word_real} | Ký tự khớp: {mapped_letters}")

        is_letter_correct = CharacterMatching.getWhichLettersWereTranscribedCorrectly(
            word_real, mapped_letters
        )

        is_letter_correct_all_words += ''.join([str(is_correct)
                                                for is_correct in is_letter_correct]) + ' '

    print("Kết quả khớp tất cả từ:", is_letter_correct_all_words)

    return is_letter_correct_all_words


def lambda_handler(event, context):
    """
    Hàm xử lý chính để nhận yêu cầu từ người dùng và thực hiện so sánh text hoặc IPA.
    """
    # Parse dữ liệu từ yêu cầu
    data = json.loads(event['body'])
    real_text = data['title']
    print("-------------")
    print(real_text)
    file_bytes = base64.b64decode(
        data['base64Audio'][22:].encode('utf-8'))
    language = data['language']
    comparison_mode = data['comparison_mode']  # 'text' hoặc 'ipa'

    if len(real_text) == 0:
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': ''
        }

    start = time.time()
    random_file_name = './' + utilsFileIO.generateRandomString() + '.ogg'
    with open(random_file_name, 'wb') as f:
        f.write(file_bytes)
    print('Time for saving binary in file:', str(time.time() - start))

    # Load và xử lý âm thanh
    audio_input, sample_rate = librosa.load(random_file_name, sr=16000)

    # Chọn mô hình và processor dựa trên chế độ so sánh
    if comparison_mode == 'text':
        processor = text_processor
        model = text_model
    elif comparison_mode == 'ipa':
        processor = ipa_processor
        model = ipa_model
    else:
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': '*',
                'Access-Control-Allow-Credentials': "true",
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'OPTIONS,POST,GET'
            },
            'body': 'Invalid comparison mode. Choose either "text" or "ipa".'
        }

    input_values = processor(audio_input, return_tensors="pt", sampling_rate=16000).input_values

    # Chuyển đầu vào và mô hình lên GPU (nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_values = input_values.to(device)
    model.to(device)

    # Thực hiện dự đoán
    with torch.no_grad():
        logits = model(input_values, return_dict=True).logits

    predicted_ids = torch.argmax(logits, axis=-1)
    transcription = processor.batch_decode(predicted_ids)

    # Xử lý kết quả
    transcription_str = ''.join(transcription).replace('[PAD]', '')

    print("Transcription:", transcription_str)

    # Thực hiện so sánh
    if comparison_mode == 'text':
        # So sánh text trực tiếp
        real_words = real_text
        predicted_words = transcription_str
        matching_result = compare_words(real_words, predicted_words)

        print(real_words)
        print(predicted_words)
        print(matching_result)
    elif comparison_mode == 'ipa':
        # So sánh IPA
        real_ipa = ipa.convert(real_text).replace("ˈ", "").split()
        predicted_ipa = transcription_str.split()
        matching_result = compare_words(real_ipa, predicted_ipa)

    # Tính toán phần trăm đúng
    matching_check = matching_result.replace(' ','')
    matching_result_str = ''.join(map(str, matching_check))  # Chuyển danh sách kết quả sang chuỗi
    print("Matching Result (string):", matching_result_str)

    # Đếm số lượng đúng (số 1) trong chuỗi matching_result
    correct = matching_result_str.count('1')
    total = len(matching_result_str)

    # Tính score
    score = (correct / total) * 100 if total > 0 else 0

    # Debug
    print("Correct:", correct)
    print("Total:", total)
    print("Score:", score)
    os.remove(random_file_name)

    # Trả về kết quả
    return {
        'real_text': real_text,
        'predicted_text': transcription_str,
        'matching_result': matching_result,
        'score': score,
        'base64_audio': data['base64Audio']
    }


