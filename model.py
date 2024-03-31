import torch

from transformers import DistilBertModel
from torch import nn, Tensor

from utils.DevConf import DevConf

class SentiDistilBert(nn.Module):
    def __init__(
            self,
            bert: DistilBertModel = None,
            dropout: float = 0.1,
            numLabels: int = 2,
            freezeBert: bool = True,
            devConf: DevConf = DevConf(),
            ):
        super(SentiDistilBert, self).__init__()
        self._bert: DistilBertModel = DistilBertModel.from_pretrained('distilbert-base-uncased').to(**devConf.__dict__) if bert is None else bert.to(**devConf.__dict__)

        if freezeBert:
            for param in self._bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout).to(**devConf.__dict__)
        
        self.linear = nn.Linear(
            self._bert.config.hidden_size,
            numLabels,
            **devConf.__dict__
            )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids: Tensor, attention_mask: Tensor, bert_no_grad: bool=True) -> Tensor:
        if bert_no_grad:
            with torch.no_grad():
                output = self._bert.forward(input_ids, attention_mask)
        else:
            output = self._bert.forward(input_ids, attention_mask)
        output = self.dropout(output.last_hidden_state[:, 0, :])
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
