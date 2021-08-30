import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
import numpy as np
import matplotlib.pyplot as plt


def main():
    #############################################################################
    ####      DATA      #########################################################
    #############################################################################
    # Model Params
    max_ht = 20
    ph = 5

    model_params = dict()
    model_params['batch_first'] = True
    model_params['input_dim'] = 1
    model_params['output_dim'] = 1
    model_params['hidden_dim'] = 32
    model_params['num_layers'] = 1
    model_params['use_input_embedder'] = True
    model_params['ph'] = ph

    # Create Data
    t_start = 0
    t_stop = 1000
    sample_frequency = 5
    num_steps = (t_stop - t_start) * sample_frequency
    num_pillars = num_steps + 1
    t = torch.linspace(t_start, t_stop, num_pillars)

    x = torch.sin(t)

    # Create Sequences
    first_idx = max_ht
    last_idx = len(x) - ph + 1
    hist_sequences = torch.stack([x[current_ts - max_ht: current_ts] for current_ts in range(first_idx, last_idx)],
                                 dim=0).unsqueeze(2)
    fut_sequences = torch.stack([x[current_ts: current_ts + ph] for current_ts in range(first_idx, last_idx)],
                                dim=0).unsqueeze(2)


    ###############################################################################
    ###     MODEL     #############################################################
    ###############################################################################
    # MODEL HYPERPARAMETERS
    encoder_params = dict()
    encoder_params['input_dim'] = 1
    encoder_params['hidden_dim'] = 32
    encoder_params['latent_dim'] = 64
    encoder_params['num_layers'] = 2
    encoder_params['batch_first'] = True

    decoder_params = dict()
    decoder_params['latent_dim'] = encoder_params['latent_dim']
    decoder_params['hidden_dim'] = 64
    decoder_params['input_dim'] = encoder_params['input_dim']
    decoder_params['output_dim'] = 1
    decoder_params['num_layers'] = 2
    decoder_params['batch_first'] = encoder_params['batch_first']
    decoder_params['ph'] = 5

    # SELECT MODEL
    # model = LSTM_many_to_one(model_params)
    model = Autoencoder(encoder_params, decoder_params)

    model.set_encoder_type('VAE')
    model.set_decoder_type('output_as_input')
    model.set_dynamics(False)
    model.display_encoder_types()
    model.display_decoder_types()
    model.display_dynamics()

    ################################################################################################
    ###     TRAINING PARAMETERS     ################################################################
    ################################################################################################
    train_params = dict()
    train_params['learning_rate'] = 0.01
    train_params['num_epochs'] = 50

    optimizer = torch.optim.Adam(model.parameters(), lr=train_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)



    #################################################################################################
    ###     TRAINING    #############################################################################
    #################################################################################################
    model.train()
    for epoch in range(train_params['num_epochs']):
        pred_sequences, KL = model(hist_sequences)
        log_p_y_xz = - MSESequenceLoss(pred_sequences, fut_sequences)
        ELBO = log_p_y_xz  # - KL
        loss = - ELBO
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if epoch % 5 == 0:
            print(f'loss:{loss.detach():.4}')


    #################################################################################################
    ###      VISUALIZE PREDICTION     ###############################################################
    #################################################################################################
    # Predict Many to Many
    model.eval()
    eval_timestep = 180
    pred_hor = ph

    sequence = x[eval_timestep - max_ht: eval_timestep].unsqueeze(0).unsqueeze(2)
    hist_sequence = sequence.clone()
    pred_sequence, kl = model(hist_sequence)
    pred_sequence = torch.cat([sequence[:, -1, :].unsqueeze(1), pred_sequence], dim=1)
    fut_sequence = x[eval_timestep: eval_timestep + pred_hor].unsqueeze(0).unsqueeze(2)
    fut_sequence = torch.cat([sequence[:, -1, :].unsqueeze(1), fut_sequence], dim=1)

    plt.plot(range(eval_timestep - max_ht, eval_timestep), hist_sequence.squeeze(), label='hist');
    plt.plot(range(eval_timestep - 1, eval_timestep + pred_hor), fut_sequence.squeeze(), label='fut');
    plt.plot(range(eval_timestep - 1, eval_timestep + pred_hor), pred_sequence.squeeze().detach(), label='pred');
    plt.legend(loc='best');


if __name__ == '__main__':
    main()