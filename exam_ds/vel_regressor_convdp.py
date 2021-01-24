import torch

# Model

class VelRegressorConvDp(torch.nn.Module):
    def __init__(self, Nin=6, Nout=1, Nlinear=5580, dropout=0.2):
        super(VelRegressorConvDp, self).__init__()

        # Convolutional layers
        self.model1 = torch.nn.Sequential(
            torch.nn.Conv1d(Nin, 180, kernel_size=1, stride=1, groups=Nin),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(180, 180, kernel_size=3, stride=1, groups=Nin),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(180, 180, kernel_size=7, stride=1),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.MaxPool1d(10, stride=6),
        )

        # Fully connected layers
        self.model2 = torch.nn.Sequential(
            torch.nn.Linear(Nlinear, 10 * 40),
            torch.nn.ReLU(),
            torch.nn.Linear(10 * 40, 100),
            torch.nn.ReLU())

        # Last FC
        self.model3 = torch.nn.Sequential(
            torch.nn.Linear(100, 3)
        )

    # Forward pass
    def forward(self, x):
        # x tensor shape (10, 6, 200) in batch_mode(10)
        x = self.model1(x)
        x = x.view(x.size(0), -1)
        x = self.model2(x)
        x = self.model3(x)
        y = torch.norm(x, dim=1)
        return y
