import os
from typing import List, Tuple
from collections import Counter
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import itertools
# Constants
LINES_PER_FILE = 1024  # Số dòng mỗi file nhỏ

class GECToRDataset:
    def __init__(
        self,
        src_file_path: str = None,
        d_labels_path: str = None,
        labels_path: str = None,
        word_masks_path: str =None,
        datapath: str = None,
        tokenizer: PreTrainedTokenizer = None,
        max_length: int = 64
    ):
        self.tokenizer = tokenizer
        self.src_file_path = src_file_path
        self.d_labels_path = d_labels_path
        self.labels_path = labels_path
        self.word_masks_path = word_masks_path
        self.max_length = max_length
        self.label2id = None
        self.d_label2id = None
        self.s_path = None
        self.dl_path = None
        self.l_path = None
        self.w_path = None
        self.datapath= datapath
        count = 0

    def load_path(self):
            self.s_path = open(self.src_file_path, 'r').readlines()
            self.dl_path = open(self.d_labels_path, 'r').readlines()
            self.l_path = open(self.labels_path, 'r').readlines()
            self.w_path = open(self.word_masks_path, 'r').readlines()

    def __len__(self):
        if self.s_path is None:
            self.load_path()
        return len(self.s_path) * LINES_PER_FILE

    def __getitem__(self, idx):
        if self.s_path is None:
            self.load_path()

        q , r = divmod(idx,LINES_PER_FILE)
        ss = open(os.path.join(self.datapath, self.s_path[q].rstrip('\n')), "r", encoding = 'utf-8').readlines()
        dls = open(os.path.join(self.datapath, self.dl_path[q].rstrip('\n')), "r", encoding = 'utf-8').readlines()
        ls = open(os.path.join(self.datapath, self.l_path[q].rstrip('\n')), "r", encoding = 'utf-8').readlines()
        ws = open(os.path.join(self.datapath, self.w_path[q].rstrip('\n')), "r", encoding = 'utf-8').readlines()

        s=ss[r].split()
        dl=[int(item) for item in dls[r].split()]
        l=[int(item) for item in ls[r].split()]
        w=[int(item) for item in ws[r].split()]

        encode = self.tokenizer(
            s,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            is_split_into_words=True
        )
        return {
            'input_ids': encode['input_ids'].squeeze(),
            'attention_mask': encode['attention_mask'].squeeze(),
            'd_labels': torch.tensor(dl).squeeze(),
            'labels': torch.tensor(l).squeeze(),
            'word_masks': torch.tensor(w).squeeze()
        }

    def load_bin(self, path):
        self.datapath=path
        self.src_file_path = os.path.join(path, 'src_paths.txt')
        self.d_labels_path = os.path.join(path, 'subword_d_labels_paths.txt')
        self.labels_path = os.path.join(path, 'subword_labels_paths.txt')
        self.word_masks_path = os.path.join(path, 'word_masks.txt')


    def append_vocab(self, label2id, d_label2id):
        self.label2id = label2id
        self.d_label2id = d_label2id
        # print(self.labels_path)

        with open(self.labels_path, 'r', encoding = 'utf-8') as labels_paths:
            for labels_path in labels_paths:
                # print(labels_path)
                labels_content = []
                with open(os.path.join(self.datapath, labels_path.rstrip('\n')), 'r', encoding = 'utf-8') as labels:
                    for label in labels:
                        # print(label)
                        labels_new=[]
                        label_split = label.split()
                        for l in label_split:
                            # print(str(self.label2id.get(l, self.label2id['<OOV>'])))
                            labels_new.append(str(self.label2id.get(l, self.label2id['<OOV>'])))
                        labels_content.append(' '.join(labels_new))
                        
                with open(os.path.join(self.datapath, labels_path.rstrip('\n')), 'wb') as labels_file:
                    for line in labels_content:
                        labels_file.write(f"{line}\n".encode('utf-8'))

        with open(self.d_labels_path, 'r', encoding = 'utf-8') as d_labels_paths:
            for d_labels_path in d_labels_paths:
                d_labels_content = []
                with open(os.path.join(self.datapath, d_labels_path.rstrip('\n')), 'r', encoding = 'utf-8') as d_labels:
                    for d_label in d_labels:
                        # print(d_label)
                        d_labels_new=[]
                        d_label_split=d_label.split()
                        for l in d_label_split:
                            d_labels_new.append(str(self.d_label2id[l]))
                        d_labels_content.append(' '.join(d_labels_new))
                
                with open(os.path.join(self.datapath, d_labels_path.rstrip('\n')), "wb") as d_labels_file:
                    for line in d_labels_content:
                        d_labels_file.write(f"{line}\n".encode('utf-8'))
                    

    
    def get_labels_freq(self, exluded_labels: List[str] = []):
        assert(self.labels_path is not None and self.d_labels_path is not None)
        flatten_labels=[]
        with open(self.labels_path, 'r', encoding = 'utf-8') as labels_path:
            for path in labels_path:
                with open(path, 'r', encoding = 'utf-8') as labels:
                    for label in labels:
                        label_split = label.split()
                        for l in label_split:
                            if l not in exluded_labels:
                                flatten_labels.append(l)
        flatten_d_labels=[]
        with open(self.d_labels_path, 'r', encoding = 'utf-8') as d_labels_path:
            for path in d_labels_path:
                with open(path, 'r', encoding = 'utf-8') as d_labels:
                    for d_label in d_labels:
                        d_label_split = d_label.split()
                        for d_l in d_label_split:
                            if d_l not in exluded_labels:
                                flatten_d_labels.append(d_l)

        return Counter(flatten_labels), Counter(flatten_d_labels)

# Hàm lưu file vào thư mục cụ thể với tối đa LINES_PER_FILE dòng
def save_to_binary_file(data: List[str], output_dir: str, prefix: str, file_idx: int):
    file_path = os.path.join(output_dir, prefix)
    os.makedirs(file_path, exist_ok=True)
    file_path = os.path.join(file_path, f"{file_idx}.bin")
    with open(file_path, "wb") as f:
        for line in data:
            f.write(f"{line}\n".encode("utf-8"))
    from posixpath import join
    path= join(prefix, f"{file_idx}.bin")
    # path= os.path.join(prefix, f"{file_idx}.bin")
    return path


# Chỉnh sửa hàm load_gector_format để lưu vào các thư mục nhỏ
def load_gector_format(input_file: str, delimeter: str = 'SEPL|||SEPR', additional_delimeter: str = 'SEPL__SEPR') -> Tuple[str, str]:
    base_output_dir = "bin"
    base_output_dir = os.path.join(base_output_dir, input_file.split('.')[0].split('/')[-1])
    os.makedirs(base_output_dir, exist_ok=True)
    src_output_dir = os.path.join(base_output_dir, 'src')
    os.makedirs(src_output_dir, exist_ok=True)
    labels_output_dir = os.path.join(base_output_dir, 'labels')
    os.makedirs(labels_output_dir, exist_ok=True)

    src_lines = []
    label_lines = []
    file_idx = 0
    src_paths = []
    label_paths = []
    
    # Đọc file nguồn và tách thành các phần tử src và labels
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            src = [x.split(delimeter)[0] for x in line.split()]
            labels = [x.split(delimeter)[1].split(additional_delimeter)[0] for x in line.split()]
            src_lines.append(" ".join(src))
            label_lines.append(" ".join(labels))

            if len(src_lines) == LINES_PER_FILE:
                path = save_to_binary_file(src_lines, base_output_dir, 'src', file_idx)
                src_paths.append(path)
                src_lines = []
                path = save_to_binary_file(label_lines, base_output_dir, 'labels', file_idx)
                label_paths.append(path)
                # print(label_lines)
                label_lines = []
                file_idx += 1

    # Save list of paths
    src_paths_file = os.path.join(base_output_dir, f"src_paths.txt")
    with open(src_paths_file, "w") as f:
        f.write("\n".join(src_paths))

    labels_paths_file = os.path.join(base_output_dir, f"labels_paths.txt")
    with open(labels_paths_file, "w") as f:
        f.write("\n".join(label_paths))

    return src_paths_file, labels_paths_file, base_output_dir

# Chỉnh sửa hàm align_labels_to_subwords để lưu vào từng thư mục nhỏ cho các loại dữ liệu khác
def align_labels_to_subwords(
    input_file: str,
    src_paths_file: str,
    labels_paths_file: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    keep_label: str = '$KEEP',
    pad_token: str = '<PAD>',
    correct_label: str = '$CORRECT',
    incorrect_label: str = '$INCORRECT'
):
    base_output_dir = "bin"
    base_output_dir = os.path.join(base_output_dir, input_file.split('.')[0].split('/')[-1])

    subword_labels_paths = []
    subword_d_labels_paths = []
    word_masks_paths = []
    
    file_idx = 0
    # for i in tqdm(itr):
    with open(src_paths_file, 'r') as src_file_paths, open(labels_paths_file, 'r') as labels_file_paths:
        for src_file_path, labels_file_path in itertools.zip_longest(src_file_paths, labels_file_paths):
            batch_srcs = []
            with open(os.path.join(base_output_dir, src_file_path.rstrip('\n')), 'r', encoding='utf-8') as srcs:
                for src in srcs:
                    batch_srcs.append(src.split())
    
        # for labels_file_path in labels_file_paths:
            batch_word_labels = []
            count = 0
            with open(os.path.join(base_output_dir, labels_file_path.rstrip('\n')), 'r', encoding='utf-8') as labels:
                for label in labels:
                    batch_word_labels.append(label.split())
                    # print(batch_word_labels)


            encode = tokenizer(
                batch_srcs,
                return_tensors='pt',
                max_length=max_length,
                padding='max_length',
                truncation=True,
                is_split_into_words=True
            )
            subword_d_labels = []
            subword_labels = []
            word_masks = []
            for j, wlabels in enumerate(batch_word_labels):
                d_labels = []
                labels = []
                wmask = []
                word_ids = encode.word_ids(j)

                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        labels.append(pad_token)
                        d_labels.append(pad_token)
                        wmask.append('0')
                    elif word_idx != previous_word_idx:
                        l = wlabels[word_idx]  # Lấy nhãn từ dòng tương ứng
                        labels.append(l)
                        wmask.append('1')
                        if l != keep_label:
                            d_labels.append(incorrect_label)
                        else:
                            d_labels.append(correct_label)
                    else:
                        labels.append(pad_token)
                        d_labels.append(pad_token)
                        wmask.append('0')
                    previous_word_idx = word_idx
                subword_d_labels.append(" ".join(d_labels))
                subword_labels.append(" ".join(labels))
                word_masks.append(" ".join(wmask))

    # Lưu từng file vào thư mục riêng
            subword_labels_path = save_to_binary_file(subword_labels, base_output_dir, "subword_labels", file_idx)
            subword_d_labels_path = save_to_binary_file(subword_d_labels, base_output_dir, "subword_d_labels", file_idx)
            word_masks_path = save_to_binary_file(word_masks, base_output_dir, "word_masks", file_idx)
            subword_labels_paths.append(subword_labels_path)
            subword_d_labels_paths.append(subword_d_labels_path)
            word_masks_paths.append(word_masks_path)
            file_idx += 1

    subword_labels_output_dir = os.path.join(base_output_dir, f"subword_labels_paths.txt")
    with open(subword_labels_output_dir, "w") as f:
        f.write("\n".join(subword_labels_paths))

    subword_d_labels_output_dir = os.path.join(base_output_dir, f"subword_d_labels_paths.txt")
    with open(subword_d_labels_output_dir, "w") as f:
        f.write("\n".join(subword_d_labels_paths))

    word_masks_output_dir = os.path.join(base_output_dir, f"word_masks.txt")
    with open(word_masks_output_dir, "w") as f:
        f.write("\n".join(word_masks_paths))

    return (
        subword_d_labels_output_dir,
        subword_labels_output_dir,
        word_masks_output_dir
    )

def load_dataset(
    input_file: str,
    tokenizer: PreTrainedTokenizer,
    delimeter: str = 'SEPL|||SEPR',
    additional_delimeter: str = 'SEPL__SEPR',
    max_length: int = 128
):
    src_paths_file, labels_paths_file , base_output_dir= load_gector_format(
        input_file,
        delimeter=delimeter,
        additional_delimeter=additional_delimeter
    )

    subword_d_labels_path, subword_labels_path, word_masks_path = align_labels_to_subwords(
        input_file,
        src_paths_file,
        labels_paths_file,
        tokenizer=tokenizer,
        max_length=max_length
    )
    return GECToRDataset(
        src_file_path=src_paths_file,
        d_labels_path=subword_d_labels_path,
        labels_path=subword_labels_path,
        word_masks_path=word_masks_path,
        datapath=base_output_dir,
        tokenizer=tokenizer,
        max_length=max_length
    )
