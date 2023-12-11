# Structure list


## Structure 1

```python
class Net(nn.Module):
    def __init__(self, hidden=512, features=35, crossing=False):
        super(Net, self).__init__()
  
        self.crossing = crossing
        if crossing:
            input_dim = features**2
        else:
            input_dim = features
        self.features = features
  
        self.input_proj = nn.Linear(input_dim, hidden)
        self.l1 = nn.Linear(hidden, hidden*4)
        self.l2 = nn.Linear(hidden*4, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.constant_(self.l2.weight, 0)
        nn.init.constant_(self.head.weight, 0)

    def forward(self, x):
        h = x
        if self.crossing:
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                h = x.unsqueeze(2) @ x.unsqueeze(1)
                h[:, torch.eye(self.features, device=x.device).bool()] = x
            h = h.flatten(1)
        x = self.norm(self.input_proj(h))
        h = F.mish(self.l1(x))
        h = self.l2(h)
        h = self.head(h + x)
        return h
```
