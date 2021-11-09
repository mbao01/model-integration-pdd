import torch
import nltk
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

print(torch.__version__)

if torch.cuda.is_available():
    weights = torch.load('./data/model.h5')
else:
    weights = torch.load('./data/model_cpu.h5')

class RecommenderNet(nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        super(RecommenderNet, self).__init__()
        self.e = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * 50 * 2, 512)
        self.hidden = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.linear2 = nn.Linear(128, 1)

    def forward(self, s1, s2):
        e1 = self.e(s1)
        e2 = self.e(s2)
        e3 = torch.cat([e1, e2], 1)
        ersh = e3.shape
        e3 = torch.reshape(e3, (ersh[0], ersh[1] * ersh[2]))
        out = F.relu(self.linear1(e3))
        out = self.hidden(out)
        out = torch.sigmoid(self.linear2(out))
        return out

    def _init(self):
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)

class Predictor:
    def __init__(self):
        vocab_size = 16245
        embedding_dim = 30
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net = RecommenderNet(vocab_size, embedding_dim).to(self.device)
        net.load_state_dict(weights)
        self.net = net
        with open('./data/word_to_ix.pickle', 'rb') as input_file:
            self.word_to_ix = pickle.load(input_file)
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

    def predict(self, puzzle, issues):
        with torch.no_grad():
            sentense = nltk.tokenize.word_tokenize(puzzle)
            len_v = min(len(sentense), 50)
            vector = [self.word_to_ix.get(w, 0) for w in sentense][:len_v]
            vector_puzzle = torch.tensor(np.concatenate([vector, np.zeros(50 - len_v)], axis=0).astype(np.int32))
            stest__ = []
            for p in issues:
                # p = get_text(p)
                sentense = nltk.tokenize.word_tokenize(p)
                len_v = min(50, len(sentense))
                vector = [self.word_to_ix.get(w, 0) for w in sentense][:len_v]
                v = np.concatenate([vector, np.zeros(50 - len_v)], axis=0).astype(np.int32)
                stest__.append(torch.tensor(v))

            out = self.net(vector_puzzle.repeat(len(stest__), 1).to(self.device),
                           torch.stack(stest__).to(self.device)).squeeze()
            return sorted(zip([i for i in issues], out.cpu().detach().numpy()), key=lambda x: -x[1])

def get_text(issue):
    return ' '.join([elem for elem in [issue.title, issue.body] if elem])
