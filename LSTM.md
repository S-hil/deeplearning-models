

```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.io import *
from fastai.conv_learner import *

from fastai.column_data import *
```

## Setup


```python
PATH='data/nietzsche/'
```


```python
text = open(f'{PATH}nietzsche.txt').read()
print('corpus length:', len(text))
```

    corpus length: 40000



```python
text[:400]
```




    'PREFACE\n\n\nSUPPOSING that Truth is a woman--what then? Is there not ground\nfor suspecting that all philosophers, in so far as they have been\ndogmatists, have failed to understand women--that the terrible\nseriousness and clumsy importunity with which they have usually paid\ntheir addresses to Truth, have been unskilled and unseemly methods for\nwinning a woman? Certainly she has never allowed herself '




```python
chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)
```

    total chars: 76


Sometimes it's useful to have a zero value in the dataset, e.g. for padding


```python
chars.insert(0, "\0")

''.join(chars[1:-6])
```




    '\n !"\'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXY_abcdefghijklmnopqrst'




```python
char_indices = {c: i for i, c in enumerate(chars)}
indices_char = {i: c for i, c in enumerate(chars)}
```


```python
idx = [char_indices[c] for c in text]

idx[:10]
```




    [39, 41, 28, 29, 24, 26, 28, 1, 1, 1]




```python
''.join(indices_char[i] for i in idx[:70])
```




    'PREFACE\n\n\nSUPPOSING that Truth is a woman--what then? Is there not gro'



## Three char model

### Create inputs


```python
cs=3
c1_dat = [idx[i]   for i in range(0, len(idx)-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-cs, cs)]
```

Our inputs


```python
x1 = np.stack(c1_dat)
x2 = np.stack(c2_dat)
x3 = np.stack(c3_dat)
```

Our output


```python
y = np.stack(c4_dat)
```


```python
x1.shape, y.shape
```




    ((13333,), (13333,))



### Create and train model


```python
n_hidden = 256
```


```python
n_fac = 42
```


```python
class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)

        # The 'green arrow' from our diagram - the layer operation from input to hidden
        self.l_in = nn.Linear(n_fac, n_hidden)

        # The 'orange arrow' from our diagram - the layer operation from hidden to hidden
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        
        # The 'blue arrow' from our diagram - the layer operation from hidden to output
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, c1, c2, c3):
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))
        
        h = V(torch.zeros(in1.size()))
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))
```


```python
md = ColumnarModelData.from_arrays('.', [-1], np.stack([x1,x2,x3], axis=1), y, bs=512)
```


```python
m = Char3Model(vocab_size, n_fac)
```


```python
it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))
```


```python
opt = optim.Adam(m.parameters(), 1e-2)
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.911144   3.086874  
    





    [array([3.08687])]




```python
set_lrs(opt, 0.001)
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.84525  6.52312]                                 
    


### Test model


```python
def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]
```


```python
get_next('y. ')
```




    'p'




```python
get_next('ppl')
```




    'e'




```python
get_next(' th')
```




    'e'




```python
get_next('and')
```




    ' '



## Our first RNN!

### Create inputs


```python
cs=8
```


```python
c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(len(idx)-cs)]
```


```python
c_out_dat = [idx[j+cs] for j in range(len(idx)-cs)]
```


```python
xs = np.stack(c_in_dat, axis=0)
```


```python
xs.shape
```




    (39992, 8)




```python
y = np.stack(c_out_dat)
```

### Create and train model


```python
val_idx = get_cv_idxs(len(idx)-cs-1)
```


```python
md = ColumnarModelData.from_arrays('.', val_idx, xs, y, bs=512)
```


```python
class CharLoopModel(nn.Module):
    # This is an RNN!
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden))
        for c in cs:
            inp = F.relu(self.l_in(self.e(c)))
            h = F.tanh(self.l_hidden(h+inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)
```


```python
m = CharLoopModel(vocab_size, n_fac)
opt = optim.Adam(m.parameters(), 1e-2)
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.540714   2.313722  
    





    [array([2.31372])]




```python
set_lrs(opt, 0.001)
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.73588  1.75103]                                 
    



```python
class CharLoopConcatModel(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac+n_hidden, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden))
        for c in cs:
            inp = torch.cat((h, self.e(c)), 1)
            inp = F.relu(self.l_in(inp))
            h = F.tanh(self.l_hidden(inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)
```


```python
m = CharLoopConcatModel(vocab_size, n_fac)
opt = optim.Adam(m.parameters(), 1e-3)
```


```python
it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.955463   2.684819  
    





    [array([2.68482])]




```python
set_lrs(opt, 1e-4)
```


```python
fit(m, md, 1, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.69008  1.69936]                                 
    


### Test model


```python
def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]
```


```python
get_next('for thos')
```




    ' '




```python
get_next('part of ')
```




    't'




```python
get_next('queens a')
```




    'n'



## RNN with pytorch


```python
class CharRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        
        return F.log_softmax(self.l_out(outp[-1]), dim=-1)
```


```python
m = CharRnn(vocab_size, n_fac)
opt = optim.Adam(m.parameters(), 1e-3)
```


```python
it = iter(md.trn_dl)
*xs,yt = next(it)
```


```python
t = m.e(V(torch.stack(xs)))
t.size()
```




    torch.Size([8, 512, 42])




```python
ht = V(torch.zeros(1, 512,n_hidden))
outp, hn = m.rnn(t, ht)
outp.size(), hn.size()
```




    (torch.Size([8, 512, 256]), torch.Size([1, 512, 256]))




```python
t = m(*V(xs)); t.size()
```




    torch.Size([512, 76])




```python
fit(m, md, 4, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.86065  1.84255]                                 
    [ 1.       1.68014  1.67387]                                 
    [ 2.       1.58828  1.59169]                                 
    [ 3.       1.52989  1.54942]                                 
    



```python
set_lrs(opt, 1e-4)
```


```python
fit(m, md, 2, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=2), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.92928    2.668219  
        1      2.611671   2.442238                            
    





    [array([2.44224])]



### Test model


```python
def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]
```


```python
get_next('for thos')
```




    ' '




```python
def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res
```


```python
get_next_n('for thos', 40)
```




    'for thos the the the the the the the the the the'



## Multi-output model

### Setup

Let's take non-overlapping sets of characters this time


```python
c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(0, len(idx)-cs-1, cs)]
```

Then create the exact same thing, offset by 1, as our labels


```python
c_out_dat = [[idx[i+j] for i in range(cs)] for j in range(1, len(idx)-cs, cs)]
```


```python
xs = np.stack(c_in_dat)
xs.shape
```




    (4999, 8)




```python
ys = np.stack(c_out_dat)
ys.shape
```




    (4999, 8)




```python
xs[:cs,:cs]
```




    array([[39, 41, 28, 29, 24, 26, 28,  1],
           [ 1,  1, 42, 44, 39, 39, 38, 42],
           [32, 37, 30,  2, 69, 57, 50, 69],
           [ 2, 43, 67, 70, 69, 57,  2, 58],
           [68,  2, 50,  2, 72, 64, 62, 50],
           [63,  9,  9, 72, 57, 50, 69,  2],
           [69, 57, 54, 63, 23,  2, 32, 68],
           [ 2, 69, 57, 54, 67, 54,  2, 63]])




```python
ys[:cs,:cs]
```




    array([[41, 28, 29, 24, 26, 28,  1,  1],
           [ 1, 42, 44, 39, 39, 38, 42, 32],
           [37, 30,  2, 69, 57, 50, 69,  2],
           [43, 67, 70, 69, 57,  2, 58, 68],
           [ 2, 50,  2, 72, 64, 62, 50, 63],
           [ 9,  9, 72, 57, 50, 69,  2, 69],
           [57, 54, 63, 23,  2, 32, 68,  2],
           [69, 57, 54, 67, 54,  2, 63, 64]])



### Create and train model


```python
val_idx = get_cv_idxs(len(xs)-cs-1)
```


```python
md = ColumnarModelData.from_arrays('.', val_idx, xs, ys, bs=512)
```


```python
class CharSeqRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        return F.log_softmax(self.l_out(outp), dim=-1)
```


```python
m = CharSeqRnn(vocab_size, n_fac)
opt = optim.Adam(m.parameters(), 1e-3)
```


```python
it = iter(md.trn_dl)
*xst,yt = next(it)
```


```python
def nll_loss_seq(inp, targ):
    sl,bs,nh = inp.size()
    targ = targ.transpose(0,1).contiguous().view(-1)
    return F.nll_loss(inp.view(-1,nh), targ)
```


```python
fit(m, md, 4, opt, nll_loss_seq)
```


    A Jupyter Widget


    [ 0.       2.59241  2.40251]                                
    [ 1.       2.28474  2.19859]                                
    [ 2.       2.13883  2.08836]                                
    [ 3.       2.04892  2.01564]                                
    



```python
set_lrs(opt, 1e-4)
```


```python
fit(m, md, 1, opt, nll_loss_seq)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                          
        0      4.32723    4.291134  
    





    [array([4.29113])]



### Identity init!


```python
m = CharSeqRnn(vocab_size, n_fac)
opt = optim.Adam(m.parameters(), 1e-2)
```


```python
m.rnn.weight_hh_l0.data.copy_(torch.eye(n_hidden))
```




    
        1     0     0  ...      0     0     0
        0     1     0  ...      0     0     0
        0     0     1  ...      0     0     0
           ...          ⋱          ...       
        0     0     0  ...      1     0     0
        0     0     0  ...      0     1     0
        0     0     0  ...      0     0     1
    [torch.FloatTensor of size 256x256]




```python
fit(m, md, 4, opt, nll_loss_seq)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=4), HTML(value='')))


    epoch      trn_loss   val_loss                          
        0      3.490897   3.1025    
        1      3.235987   2.869533                          
        2      3.063832   2.690163                          
        3      2.929215   2.572571                          
    





    [array([2.57257])]




```python
set_lrs(opt, 1e-3)
```


```python
fit(m, md, 4, opt, nll_loss_seq)
```


    A Jupyter Widget


    [ 0.       1.84035  1.85742]                                
    [ 1.       1.82896  1.84887]                                
    [ 2.       1.81879  1.84281]                               
    [ 3.       1.81337  1.83801]                                
    


## Stateful model

### Setup


```python
from torchtext import vocab, data

from fastai.nlp import *
from fastai.lm_rnn import *

PATH='data/nietzsche/'

TRN_PATH = 'trn/'
VAL_PATH = 'val/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

%ls {PATH}
```

    nietzsche.txt       nietzschesmall.txt  [34mtrn[m[m/                [34mval[m[m/



```python
%ls {PATH}trn
```

    nietzschesmall.txt



```python
TEXT = data.Field(lower=True, tokenize=list)
bs=64; bptt=8; n_fac=42; n_hidden=256

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=3)

len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)
```




    (75, 45, 1, 39383)



### RNN


```python
class CharSeqStatefulRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        self.vocab_size = vocab_size
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))
```


```python
m = CharSeqStatefulRnn(md.nt, n_fac, 512)
opt = optim.Adam(m.parameters(), 1e-3)
```


```python
fit(m, md, 4, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.81983  1.81247]                                 
    [ 1.       1.63097  1.66228]                                 
    [ 2.       1.54433  1.57824]                                 
    [ 3.       1.48563  1.54505]                                 
    



```python
set_lrs(opt, 1e-4)

fit(m, md, 1, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      3.223129   2.929459  
    





    [array([2.92946])]



### RNN loop


```python
# From the pytorch source

def RNNCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    return F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
```


```python
class CharSeqStatefulRnn2(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNNCell(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp = []
        o = self.h
        for c in cs: 
            o = self.rnn(self.e(c), o)
            outp.append(o)
        outp = self.l_out(torch.stack(outp))
        self.h = repackage_var(o)
        return F.log_softmax(outp, dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))
```


```python
m = CharSeqStatefulRnn2(md.nt, n_fac, 512)
opt = optim.Adam(m.parameters(), 1e-3)
```


```python
fit(m, md, 4, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=4), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.634824   2.435599  
        1      2.397607   2.288882                            
        2      2.261155   2.169498                            
        3      2.158415   2.082467                            
    





    [array([2.08247])]



### GRU


```python
class CharSeqStatefulGRU(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.GRU(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))
```


```python
# From the pytorch source code - for reference

def GRUCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    return newgate + inputgate * (hidden - newgate)
```


```python
m = CharSeqStatefulGRU(md.nt, n_fac, 512)

opt = optim.Adam(m.parameters(), 1e-3)
```


```python
fit(m, md, 6, opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=6), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.667542   2.425069  
        1      2.371198   2.23098                             
        2      2.207084   2.096106                            
        3      2.081467   1.982581                            
        4      1.971633   1.877548                            
        5      1.877026   1.794282                            
    





    [array([1.79428])]




```python
set_lrs(opt, 1e-4)
```


```python
fit(m, md, 3, opt, F.nll_loss)
```


    A Jupyter Widget


    [ 0.       1.22708  1.36926]                                 
    [ 1.       1.21948  1.3696 ]                                 
    [ 2.       1.22541  1.36969]                                 
    


### Putting it all together: LSTM


```python
from fastai import sgdr

n_hidden=512
```


```python
class CharSeqStatefulLSTM(nn.Module):
    def __init__(self, vocab_size, n_fac, bs, nl):
        super().__init__()
        self.vocab_size,self.nl = vocab_size,nl
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.LSTM(n_fac, n_hidden, nl, dropout=0.5)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs):
        self.h = (V(torch.zeros(self.nl, bs, n_hidden)),
                  V(torch.zeros(self.nl, bs, n_hidden)))
```


```python
m = CharSeqStatefulLSTM(md.nt, n_fac, 512, 2)
lo = LayerOptimizer(optim.Adam, m, 1e-2, 1e-5)
```


```python
os.makedirs(f'{PATH}models', exist_ok=True)
```


```python
fit(m, md, 2, lo.opt, F.nll_loss)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=2), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      2.519296   2.222408  
        1      2.186267   1.941648                            
    





    [array([1.94165])]




```python
on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
fit(m, md, 1, lo.opt, F.nll_loss, callbacks=cb)
```


    HBox(children=(IntProgress(value=0, description='Epoch', max=1), HTML(value='')))


    epoch      trn_loss   val_loss                            
        0      1.616982   1.337493  





    [array([1.33749])]




```python
on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
fit(m, md, 2**6-1, lo.opt, F.nll_loss, callbacks=cb)
```


    A Jupyter Widget


    [ 0.       1.46053  1.43462]                                 
    [ 1.       1.51537  1.47747]                                 
    [ 2.       1.39208  1.38293]                                 
    [ 3.       1.53056  1.49371]                                 
    [ 4.       1.46812  1.43389]                                 
    [ 5.       1.37624  1.37523]                                 
    [ 6.       1.3173   1.34022]                                 
    [ 7.       1.51783  1.47554]                                 
    [ 8.       1.4921   1.45785]                                 
    [ 9.       1.44843  1.42215]                                 
    [ 10.        1.40948   1.40858]                              
    [ 11.        1.37098   1.36648]                              
    [ 12.        1.32255   1.33842]                              
    [ 13.        1.28243   1.31106]                              
    [ 14.        1.25031   1.2918 ]                              
    [ 15.        1.49236   1.45316]                              
    [ 16.        1.46041   1.43622]                              
    [ 17.        1.45043   1.4498 ]                              
    [ 18.        1.43331   1.41297]                              
    [ 19.        1.43841   1.41704]                              
    [ 20.        1.41536   1.40521]                              
    [ 21.        1.39829   1.37656]                              
    [ 22.        1.37001   1.36891]                              
    [ 23.        1.35469   1.35909]                              
    [ 24.        1.32202   1.34228]                              
    [ 25.        1.29972   1.32256]                              
    [ 26.        1.28007   1.30903]                              
    [ 27.        1.24503   1.29125]                              
    [ 28.        1.22261   1.28316]                              
    [ 29.        1.20563   1.27397]                              
    [ 30.        1.18764   1.27178]                              
    [ 31.        1.18114   1.26694]                              
    [ 32.        1.44344   1.42405]                              
    [ 33.        1.43344   1.41616]                              
    [ 34.        1.4346    1.40442]                              
    [ 35.        1.42152   1.41359]                              
    [ 36.        1.42072   1.40835]                              
    [ 37.        1.41732   1.40498]                              
    [ 38.        1.41268   1.395  ]                              
    [ 39.        1.40725   1.39433]                              
    [ 40.        1.40181   1.39864]                              
    [ 41.        1.38621   1.37549]                              
    [ 42.        1.3838    1.38587]                              
    [ 43.        1.37644   1.37118]                              
    [ 44.        1.36287   1.36211]                              
    [ 45.        1.35942   1.36145]                              
    [ 46.        1.34712   1.34924]                              
    [ 47.        1.32994   1.34884]                              
    [ 48.        1.32788   1.33387]                              
    [ 49.        1.31553   1.342  ]                              
    [ 50.        1.30088   1.32435]                              
    [ 51.        1.28446   1.31166]                              
    [ 52.        1.27058   1.30807]                              
    [ 53.        1.26271   1.29935]                              
    [ 54.        1.24351   1.28942]                              
    [ 55.        1.23119   1.2838 ]                              
    [ 56.        1.2086    1.28364]                              
    [ 57.        1.19742   1.27375]                              
    [ 58.        1.18127   1.26758]                              
    [ 59.        1.17475   1.26858]                              
    [ 60.        1.15349   1.25999]                              
    [ 61.        1.14718   1.25779]                              
    [ 62.        1.13174   1.2524 ]                              
    


### Test


```python
def get_next(inp):
    idxs = TEXT.numericalize(inp,device=-1)
    p = m(VV(idxs.transpose(0,1)))
    r = torch.multinomial(p[-1].exp(), 1)
    return TEXT.vocab.itos[to_np(r)[0]]
```


```python
get_next('for thos')
```




    'a'




```python
def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res
```


```python
print(get_next_n('for thos', 400))
```

    for thosal hunscholarigines--ors antion  of that the undressent--and asturebpessomethertogether; and wills. got mo all,watherein perhaps-begaprediced than it will to uppersion icceptameths of the truented term stylless belief your, the subtle<unk> a do long amatiour one could words to cognize of the philosophy, his fain bus-doorg as lesapertion in in shhope conscial or "invermancertay they awayge part of his 

