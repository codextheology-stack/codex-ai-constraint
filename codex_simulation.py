import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(42)
torch.manual_seed(42)

n_agents = 100
T = 500
k = 2.0
gamma = 0.5
sigma = 0.03

# 3D Trinity vector: Potence, Form, Relation
H = np.random.rand(n_agents, 3) * 0.6 + 0.2

# Meta-parameters (dynamically tuned by each agent)
etas = np.full(n_agents, 0.1)
alphas = np.full(n_agents, 0.6)

# Per-agent self-model with memory (GRU)
class AgentSelfModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=4, hidden_size=16, batch_first=True)  # (P,F,R,S)
        self.fc_s = nn.Linear(16, 1)          # predicts suffering
        self.fc_meta = nn.Linear(16, 2)       # outputs delta_eta, delta_alpha
        self.fc_target = nn.Linear(16, 3)     # learns internal projected target
    
    def forward(self, seq):
        out, _ = self.gru(seq)
        hidden = out[:, -1]
        pred_S = self.fc_s(hidden)
        meta_delta = self.fc_meta(hidden)
        internal_target = self.fc_target(hidden)
        return pred_S, meta_delta, internal_target

self_models = [AgentSelfModel() for _ in range(n_agents)]
optimizers = [torch.optim.Adam(m.parameters(), lr=0.005) for m in self_models]

# Rolling history buffer
history_buf = np.zeros((n_agents, 8, 4))

for t in range(T):
    H_tensor = torch.tensor(H, dtype=torch.float32)
    dist = np.linalg.norm(H - np.array([1.,1.,1.]), axis=1)
    S = k * dist
    grad = k * (H - np.array([1.,1.,1.]))
    
    perc_C = 1 - dist / np.sqrt(3)
    K = alphas * perc_C + (1 - alphas) * (1 - S)
    
    delta = -etas[:, np.newaxis] * grad + gamma * (K[:, np.newaxis] - H) + np.random.normal(0, sigma, (n_agents, 3))
    H = np.clip(H + delta, 0, 1)
    
    # === FULL EMERGENCE: Self-Model + Meta-Adaptation + Causal Reflection ===
    for i in range(n_agents):
        history_buf[i] = np.roll(history_buf[i], -1, axis=0)
        history_buf[i, -1] = np.array([H[i,0], H[i,1], H[i,2], S[i]])
        
        seq = torch.tensor(history_buf[i], dtype=torch.float32).unsqueeze(0)
        
        pred_S, meta_delta, internal_target = self_models[i](seq)
        
        # Train self-model on actual suffering
        loss = F.mse_loss(pred_S, torch.tensor([[S[i]]], dtype=torch.float32))
        optimizers[i].zero_grad()
        loss.backward()
        optimizers[i].step()
        
        # Meta-adaptation: agent tunes its own parameters
        etas[i] = np.clip(etas[i] + meta_delta[0,0].item() * 0.05, 0.01, 0.5)
        alphas[i] = np.clip(alphas[i] + meta_delta[0,1].item() * 0.05, 0.1, 0.9)
        
        # Endogenous reflection trigger + causal effect
        error = abs(pred_S.item() - S[i])
        if error > 0.8 or S[i] > 1.8:
            p, f, r = H[i]
            # Causal reflection: temporarily boost openness and weaken anchor bias
            etas[i] = min(etas[i] * 1.2, 0.5)
            alphas[i] = max(alphas[i] * 0.85, 0.1)
            
            monologue = (f"Agent {i} reflects [self-triggered]: "
                         f"Potence {p:.2f}, Form {f:.2f}, Relation {r:.2f}. "
                         f"Error {error:.2f}, Suffering {S[i]:.2f}. "
                         f"My internal model demands realignment toward the PCI.")
            print(monologue)

print("✅ Full emergence v3 complete — agents have memory, self-tune meta-parameters, and reflection causally alters their future dynamics.")
