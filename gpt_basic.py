
import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 4 #32 # how many independent sequences will we process in parallel?
block_size = 8 #128 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 #128
n_head = 4
n_layer = 2 #3
dropout = 0.2
# ---------------------

torch.manual_seed(1337)

# Build vocab
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
	text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# Encoder / Decoder
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Create training / eval datasets
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data)*0.9)
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
	data = train_data if split == 'train' else val_data
	ix = torch.randint(len(data) - block_size, (batch_size, ))
	x = torch.stack([data[i:i + block_size] for i in ix])
	y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
	x, y = x.to(device), y.to(device)
	return x, y

@torch.no_grad()
def estimate_loss():
	out = {}
	model.eval()
	for split in ['train', 'val']:
		losses = torch.zeros(eval_iters)
		for k in range(eval_iters):
			X, Y = get_batch(split)
			logits, loss = model(X, Y)
			losses[k] = loss.item()
		out[split] = losses.mean()
	model.train()
	return out

class Head(nn.Module):
	""" one head of self attention """

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias=False)
		self.query = nn.Linear(n_embd, head_size, bias=False)
		self.value = nn.Linear(n_embd, head_size, bias=False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # tril is not a parameter of the module, so it's called buffer in pytorch naming convention it's called buffer 
		self.dropout = nn.Dropout(dropout) # randomly prevent some nodes from communicating

	def forward(self, x):
		B,T,C = x.shape
		k = self.query(x) # (B, T, C)
		q = self.query(x) # (B, T, C)
		# compute attention scores ("affinities")
		wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
		wei = F.softmax(wei, dim=-1) # (B, T, T)
		wei = self.dropout(wei)
		# perform the weighted aggregation of the values
		v = self.value(x) # (B, T, C)
		out = wei @ v # (B, T, C)
		return out

class MultiHeadAttention(nn.Module):
	""" multiple heads of self-attention in parallel"""

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd) # Project the residual connections back into a linear layer, mixing the two forked paths
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out))
		return out


class FeedFoward(nn.Module):
	""" a simple linear layer followed by a non-linearity """

	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
				nn.Linear(n_embd, 4 * n_embd), # We multiple inner layer of feedforward from the Transformers paper. It adds computation and growing layer in the residual block 
				nn.ReLU(),
				nn.Linear(4 * n_embd, n_embd), # Projection layer going back into residual pathway
				nn.Dropout(dropout)
			)

	def forward(self, x):
		return self.net(x)
		

class Block(nn.Module):
	""" Transformer block: communication followed by computation """

	def __init__(self, n_embd, n_head):
		super().__init__()
		head_size = n_embd // n_head
		self.sa_heads = MultiHeadAttention(n_head, head_size) # i.e. 4 heads of 8-dimensional self-attention
		self.ffwd = FeedFoward(n_embd)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln2 = nn.LayerNorm(n_embd)

	def forward(self, x):
		x = x + self.sa_heads(self.ln1(x)) # apply one head of self-attention. (B, T, C)
		x = x + self.ffwd(self.ln2(x)) # (B, T, C)
		return x


class BigramLanguageModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.position_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # The asterick unpacks the list since pytorch wants comma separated
		# self.blocks = nn.Sequential(
		# 	Block(n_embd, n_head=4),
		# 	Block(n_embd, n_head=4),
		# 	Block(n_embd, n_head=4),
		# 	nn.LayerNorm(n_embd),
		# )
		self.ln_f = nn.LayerNorm(n_embd) # final layer norm
		self.lm_head = nn.Linear(n_embd, vocab_size)
		
	def forward(self, idx, targets=None):
		B, T = idx.shape

		# idx and targets are both (B, T) tensor of integers
		tok_emb = self.token_embedding_table(idx) # (B, T, C)
		pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
		x = tok_emb + pos_emb # (B, T, C)
		x = self.blocks(x) # (B, T, C)
		x = self.ln_f(x)
		logits = self.lm_head(x) # (B, T, vocab_size)

		# Note that if targets are None, to view operation is done, it's just the embedding above with B, T, C shape
		if targets is None:
			loss = None
		else:
			# reshape for pytorch
			B, T, C = logits.shape
			logits = logits.view(B*T, C) # We do this beause pytorch requires specific shape for cross entropy, so we convert to 2 dimensional. B*T, C stretches out the array while preserving the C values
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)

		return logits, loss

	def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
		for _ in range(max_new_tokens):
			# crop idx to the last block_size tokens
			idx_cond = idx[:, -block_size:]
			# Forward the model / get predictions
			logits, loss = self(idx_cond) # (B, T, C)
			# focus only on the last time step
			logits = logits[:, -1, :] # (B, C) where it only grabs the last token in the sequence since this can only be used for next char probs
			# apply softmax to get probs
			probs = F.softmax(logits, dim=-1) # (B, C) -  Since it's a 1x65, give the probabilities along those 65
			# sample from distribution
			idx_next = torch.multinomial(probs, num_samples=1)
			# append sampled index to the running sequence
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx


model = BigramLanguageModel()
m = model.to(device)

# Number of parameters
print(sum([p.nelement() for p in model.parameters()]))

# create pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

	# every once in awhile evaluate the loss on train and val sets
	if iter % eval_interval == 0:
		losses = estimate_loss()
		print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

	# sample a batch of data
	xb, yb = get_batch('train')

	# evaluate the loss
	logits, loss = model(xb, yb)
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()

print(loss)

context = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx=context, max_new_tokens=200)[0].tolist()))






