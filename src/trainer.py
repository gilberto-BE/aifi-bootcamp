import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchmetrics as TM
pl.utilities.seed.seed_everything(seed=42)

from pytorch_models import RecurrentNetwork


class Trainer:
    def __init__(
        self, 
        model, 
        optimizer_name='rmsprop', 
        lr=0.003, 
        loss_fn_name='mse'
        ):

        """
    def forward(self, inputs, word_ix):

        # Feedforward

        for l in range(self.num_layers):
            if l == 0:
                x = self.hidden[l](inputs)
                x = self.bnorm[l](x)
                if self.dropout is not None:
                    x = self.dropout[l](x)

                embeds = self.embedding(word_ix)
                # NOTE:
                # embeds has a shape of (batch_size, 1, embed_dim)
                # inorder to merge this change this with x, reshape this to
                # (batch_size, embed_dim)
                embeds = embeds.view(embeds.shape[0], embeds.shape[2])
                x = torch.cat((x, embeds.view(x.shape)),dim=1)

            else:
                x = self.hidden[l](x)
                x = self.bnorm[l](x)
                if self.dropout is not None:
                    x = self.dropout[l](x)
            x = F.relu(x)
        output= self.out(x)

        return output
        """

        self.model = model
        self.lr = lr
        self.optimizer_name=optimizer_name
        self.loss_fn_name = loss_fn_name
        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device}-device")

        if self.loss_fn_name == 'mse':
            self.loss_fn = nn.MSELoss()

        if self.optimizer_name.lower() == 'rmsprop': 
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), self.lr)

        elif self.optimizer_name.lower() == 'adam':
            pass

    def fit_epochs(self, train_loader, valid_loader=None, epochs=5):
        for epoch in epochs:
            self.fit_one_epoch(train_loader, valid_loader)


    def fit_one_epoch(self, train_loader, valid_loader=None):
        size = len(train_loader)
        self.model.to(self.device).train()
        for batch, data in enumerate(train_loader):
            x = data['features']
            y = data['target']
            print('x.shap1e:', x.shape)
            x, y = x.to(self.device), y.to(self.device)
            self._run_train_step(x, y, batch, size)

        if valid_loader is not None:
            with torch.no_grad():
                pass

    def evaluate(self, test_loader, batch, size):
        self.model.eval()
        for x, y in range(len(test_loader)):
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            loss += self.loss_fn(pred, y).item()
            print(f'val-loss: {loss.item()} [{batch * len(x)/{size}}]')


    def _run_train_step(self, x, y, batch, size):
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_loss.append(loss.item())
        # if batch % 100 == 0:
        print(f'loss: {loss.item()} [{batch * len(x)}/{size}]')

