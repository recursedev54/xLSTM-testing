import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.nn as nn
import torch.nn.functional as F

class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(input_size, hidden_size)

    def forward(self, input, hidden):
        hx, cx = hidden
        hx, cx = self.lstm(input, (hx, cx))
        return hx, cx

    def init_hidden(self, batch_size):
        return (torch.zeros(batch_size, self.hidden_size).to(device),
                torch.zeros(batch_size, self.hidden_size).to(device))

class GPT2WithXLSTM(nn.Module):
    def __init__(self, gpt2_model, input_size, hidden_size):
        super(GPT2WithXLSTM, self).__init__()
        self.gpt2 = gpt2_model
        self.xlstm = xLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, gpt2_model.config.vocab_size)

    def forward(self, input_ids, hidden=None):
        gpt2_outputs = self.gpt2(input_ids, return_dict=True, output_hidden_states=True)
        last_hidden_states = gpt2_outputs.hidden_states[-1]

        if hidden is None:
            batch_size = input_ids.size(0)
            hidden = self.xlstm.init_hidden(batch_size)

        xlstm_output = []
        for t in range(last_hidden_states.size(1)):
            hidden = self.xlstm(last_hidden_states[:, t, :], hidden)
            xlstm_output.append(hidden[0])

        xlstm_output = torch.stack(xlstm_output, dim=1)
        logits = self.linear(xlstm_output)
        return logits

    def generate(self, input_ids, max_length, temperature=1.0, top_k=50, top_p=0.9):
        self.eval()
        generated_ids = input_ids
        for _ in range(max_length):
            logits = self(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            next_token_logits[sorted_indices_to_remove] = -float('Inf')
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

            input_ids = torch.cat((input_ids, next_token), dim=1)
            generated_ids = torch.cat((generated_ids, next_token), dim=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return generated_ids

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer and model for GPT-2 Small (117M parameters)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Define input and hidden sizes for xLSTM
input_size = gpt2_model.config.hidden_size
hidden_size = 256  # Example hidden size for xLSTM

# Create the combined model
model = GPT2WithXLSTM(gpt2_model, input_size, hidden_size).to(device)

# Encode input text
input_text = '''HRate = Z (Tall Ask) "Press Space To Stop Recording" = '''
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate text using custom generate method
output = model.generate(input_ids, max_length=6, temperature=0.50, top_k=26, top_p=0.98)

# Decode and print the output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)

# Check the number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6:.1f}M")
