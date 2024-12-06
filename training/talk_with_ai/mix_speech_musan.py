import os
import random
from pydub import AudioSegment

# Đường dẫn đến folder speech trong musan
speech_folder = os.path.join('/home/qndat123ok/av_hubert/avhubert/preparation/musan', 'speech')

# Tạo thư mục để lưu các file mix giọng nói
output_speech_folder = os.path.join('Noise', 'speech')
os.makedirs(output_speech_folder, exist_ok=True)

# Hàm để lấy đoạn âm thanh ngẫu nhiên từ file
def get_random_clip(audio, duration=15000):
    """Lấy đoạn âm thanh ngẫu nhiên từ audio có độ dài duration (ms)."""
    if len(audio) <= duration:
        return audio  # Nếu file ngắn hơn 15 giây thì lấy cả file
    start_time = random.randint(0, len(audio) - duration)
    return audio[start_time:start_time + duration]

# Hàm mix nhiều đoạn âm thanh từ nhiều file
def mix_speech(files, duration=15000):
    """Mix ngẫu nhiên từ 2 đến 5 đoạn âm thanh từ danh sách files, mỗi đoạn có độ dài duration ms."""
    mix = AudioSegment.silent(duration=duration)  # Bắt đầu với một đoạn silence (im lặng)
    num_mix = random.randint(5, 10)  # Chọn ngẫu nhiên số lượng đoạn âm thanh để mix từ 2 đến 5
    for _ in range(num_mix):
        selected_file = random.choice(files)  # Chọn ngẫu nhiên 1 file từ danh sách
        audio = AudioSegment.from_file(selected_file)
        random_clip = get_random_clip(audio, duration=duration)
        mix = mix.overlay(random_clip)  # Mix (chồng) đoạn âm thanh vào đoạn mix ban đầu
    return mix

# Tìm tất cả các file wav trong folder speech
def get_wav_files(folder):
    """Trả về danh sách đường dẫn các file .wav từ tất cả các thư mục con."""
    wav_files = []
    for root, dirs, files in os.walk(folder):  # Duyệt qua tất cả các thư mục con
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))  # Thêm đường dẫn đầy đủ của file vào danh sách
    return wav_files

# Lấy danh sách các file speech từ musan/speech
speech_files = get_wav_files(speech_folder)
# Kiểm tra xem có đủ file để tạo 7000 file mix không
if len(speech_files) == 0:
    raise ValueError("Không có file .wav nào trong thư mục speech.")

# Tạo 7000 file mix ngẫu nhiên
num_files = 7000
clip_duration = 15000  # 15 giây = 15000 milliseconds

for i in range(2233,num_files):
    # Mix ngẫu nhiên các đoạn giọng nói từ danh sách các file
    mixed_audio = mix_speech(speech_files, duration=clip_duration)
    
    # Đặt tên file và lưu vào folder output
    output_path = os.path.join(output_speech_folder, f'mixed_speech_{i+1}.wav')
    mixed_audio.export(output_path, format='wav')

print("Hoàn thành việc tạo 7000 file giọng nói mix ngẫu nhiên.")
