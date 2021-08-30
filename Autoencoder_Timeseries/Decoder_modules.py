import torch
import torch.nn as nn


class Decoder_concat_output_latent(nn.Module):

    def __init__(self, decoder_params):
        super(Decoder_concat_output_latent, self).__init__()

        # Parameters
        self.ph = decoder_params['ph']
        self.hidden_dim = decoder_params['hidden_dim']
        self.output_dim = decoder_params['output_dim']
        self.concat_dim = decoder_params['latent_dim'] + decoder_params['output_dim']

        # Modules
        self.decoder = nn.LSTMCell(self.concat_dim,
                                   self.concat_dim)

        self.dense = nn.Linear(self.concat_dim,
                               decoder_params['output_dim'])

        self.init_pred = nn.Linear(decoder_params['input_dim'],
                                   decoder_params['output_dim'])

    def init_states(self, bs):
        h_0 = torch.zeros((bs, self.concat_dim))
        c_0 = torch.zeros((bs, self.concat_dim))

        return h_0, c_0

    def forward(self, x, z):
        bs = z.shape[0]
        h, c = self.init_states(bs)

        y = self.init_pred(x[:, -1, :])
        z = z.view(bs, -1)
        cat = torch.cat([z, y], dim=1)

        out_list = list()
        for t in range(self.ph):
            h, c = self.decoder(cat, (h, c))
            y = self.dense(h)
            cat = torch.cat([z, y], dim=1)
            out_list.append(y)

        out_all_timesteps = torch.stack(out_list, dim=1)

        return out_all_timesteps


class Decoder_repeat_latent(nn.Module):

    def __init__(self, decoder_params):
        super(Decoder_repeat_latent, self).__init__()

        self.decoder = nn.LSTM(decoder_params['latent_dim'],
                               decoder_params['hidden_dim'],
                               num_layers=decoder_params['num_layers'],
                               batch_first=decoder_params['batch_first'])
        self.dense = nn.Linear(decoder_params['hidden_dim'],
                               decoder_params['output_dim'])

        self.ph = decoder_params['ph']

    def forward(self, x, z):
        z_repeat = z.repeat(1, self.ph, 1)
        out, (h, c) = self.decoder(z_repeat)
        y_pred = self.dense(out)

        return y_pred


class Decoder_output_as_input(nn.Module):

    def __init__(self, decoder_params):
        super(Decoder_output_as_input, self).__init__()

        self.decoder_cell = nn.LSTMCell(decoder_params['latent_dim'],
                                        decoder_params['hidden_dim'])
        self.dense = nn.Linear(decoder_params['hidden_dim'],
                               decoder_params['output_dim'])

        self.ph = decoder_params['ph']
        self.hidden_dim = decoder_params['hidden_dim']

    def init_states(self, bs):
        h_0 = torch.zeros((bs, self.hidden_dim))
        c_0 = torch.zeros((bs, self.hidden_dim))
        return h_0, c_0

    def forward(self, hist, z):
        bs, _, dim = tuple(z.shape)

        h_1, c_1 = self.init_states(bs)

        out_list = list()
        x = z.view(bs, dim)
        for t in range(self.ph):
            h_1, c_1 = self.decoder_cell(x, (h_1, c_1))
            x = h_1
            out_list.append(h_1)
        out_all_timesteps = torch.stack(out_list, dim=1)

        y = self.dense(out_all_timesteps)

        return y