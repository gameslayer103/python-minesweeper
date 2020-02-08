import torch.nn as nn
import torch.optim as optim
import torch
import minesweeper

class MinesweeperModel(nn.Module):
    def __init__(self):
        # Set up the module
        super(MinesweeperModel, self).__init__()
        self.activation = nn.ReLU()

        # Set up the first layer
        self.conv1 = nn.Conv2d(1,32,kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32,32,kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(32,2,kernel_size=3, stride=1, padding=1)


    def forward(self, inVal):

        # Set up First layer
        out = self.activation(self.conv1(inVal))
        out = self.activation(self.conv2(out))
        out = self.conv3(out)
        
        return out

class MinesweeperTrainer():
    def __init__(self):

        # Set the current grid.
        self.currentGrid = None
        self.outGrid = None

        # Set up the model.
        self.model = MinesweeperModel()

        # Set up the optimizer
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.005)

    def forward(self, inGrid):
        self.currentGrid = inGrid
        print(inGrid)
        inGridTensor = torch.FloatTensor(inGrid).unsqueeze(0).unsqueeze(0)
        print(inGridTensor)
        self.outGrid = self.model(inGridTensor)
        print(self.outGrid)

    def getCell(self):
        currIndex = (0,0)
        maxVal = 0
        for row in range(len(self.currentGrid)):
            for col in range(len(self.currentGrid[0])):
                if self.outGrid.squeeze(0)[0,:].squeeze(0)[row,col] > maxVal:
                    currIndex = (row,col)
                    maxVal = self.outGrid.squeeze(0)[0,:].squeeze(0)[row,col]
        return currIndex

    def getGrid(self):
        return self.outGrid.squeeze(0).squeeze(0).clone()

    def sendResult(self, result):
        outGridTemp = self.outGrid
        result = result.long().unsqueeze(0)

        print(outGridTemp.shape)
        print(result.shape)

        self.loss = self.objective(outGridTemp, result)
        self.loss.backward()
        self.optimizer.step()
        print("Result", result)

if __name__ == "__main__":
    trainer = MinesweeperTrainer()
    minesweeper.playgame(False, trainer)


        
        