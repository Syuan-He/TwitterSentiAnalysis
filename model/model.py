import torch

from transformers import DistilBertModel, AutoModel
from torch import nn, Tensor
from einops import repeat

from utils.DevConf import DevConf
from model.MHABlock import MHABlock

class SentiDistilBert(nn.Module):
    def __init__(
            self,
            bert: nn.Module = None,
            dropout: float = 0.1,
            numLabels: int = 2,
            # freezeBert: bool = True,
            devConf: DevConf = DevConf(),
            ):
        super(SentiDistilBert, self).__init__()
        self._bert: nn.Module = DistilBertModel.from_pretrained('distilbert-base-uncased', cache_dir="/mnt/d/huggingface_cache").to(**devConf.__dict__) if bert is None else bert.to(**devConf.__dict__)
        self.Q = nn.Parameter(torch.randn(1, self._bert.config.hidden_size, **devConf.__dict__))
        self.decoder = MHABlock(self._bert.config.hidden_size, 1, dropout=dropout, device=devConf.device, dtype=devConf.dtype)

        # if freezeBert:
        #     for param in self._bert.parameters():
        #         param.requires_grad = False

        self.dropout = nn.Dropout(dropout).to(**devConf.__dict__)
        
        self.linear = nn.Linear(
            self._bert.config.hidden_size,
            numLabels,
            **devConf.__dict__
            )
        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def __init__(
            self,
            path: str,
            dropout: float = 0.1,
            numLabels: int = 2,
            devConf: DevConf = DevConf(),
    ):
        super(SentiDistilBert, self).__init__()
        bert: nn.Module = AutoModel.from_pretrained(path, cache_dir="/mnt/d/huggingface_cache").to(**devConf.__dict__)
        self.Q = nn.Parameter(torch.randn(1, self._bert.config.hidden_size, **devConf.__dict__))
        self.decoder = MHABlock(self._bert.config.hidden_size, 1, dropout=dropout, device=devConf.device, dtype=devConf.dtype)

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
        output = self.dropout(output.last_hidden_state)
        batch = output.shape[0]
        output, _ = self.decoder.forward(repeat(self.Q, "l d -> b l d", b=batch), output, output)
        output = output.squeeze(1)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output
