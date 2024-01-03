import torch
import torch.nn as nn
import torch.optim as optim
import math

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model, nhead, d_hid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

# 예시 데이터셋 로드 및 토크나이징
# 데이터 준비 및 배치 생성 코드...

# 모델 초기화 및 하이퍼파라미터 설정
# ntokens = len(vocab) # 예: 어휘 사전 크기
# d_model = 512  # 임베딩 차원
# nhead = 8  # 멀티헤드 어텐션의 헤드 수
# d_hid = 2048  # 피드포워드 네트워크의 차원
# nlayers = 6  # 인코더 레이어 수
# dropout = 0.1  # 드롭아웃 비율

# model = TransformerModel(ntokens, d_model, nhead, d_hid, nlayers, dropout)

# 훈련 루프
# 손실 함수 및 옵티마이저 설정
# 에포크별 훈련 및 검증...


def plot_attention(attention, sentence, predicted_sentence):
    fig, ax = plt.subplots(figsize=(10,10))
    attention = attention.squeeze(0).cpu().detach().numpy()

    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

# 예시 입력 및 모델 예측
sentence = ["이것은", "예시", "문장입니다"]
predicted_sentence = ["This", "is", "an", "example", "sentence"]
attention = torch.rand(5, 3) # 임시 어텐션 가중치

plot_attention(attention, sentence, predicted_sentence)
