import torch.nn as nn
import torch.nn.functional as F
import torch


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv1d(1, 200, 4, stride=3),
            nn.ReLU(),
            nn.Conv1d(200, 200, 5, stride=1),
            nn.ReLU(),
            # nn.Conv1d(64, 128, 3,stride=3),
            # nn.PReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.fc = nn.Sequential(
            nn.Linear(99400, 200),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.Dropout(p=0.05),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.05),
            nn.Linear(50, 8)
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)  # 根据 前一个数据 -1 自动推断维数
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output

    def get_embedding(self, x):
        return self.forward(x)


class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        print(output.size())
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)
