import collections
import numpy as np
import torch


def optimization_step(loss, accumulated_gradients, num_accumulated_gradients, optimizer,
                      current_hs, current_cs, update_every, num_bees, hidden_size):
    if loss is not None:
        loss.backward()

        # all parameters which have gradients
        grad_params = filter(lambda p: p.grad is not None, optimizer.param_groups[0]['params'])
        # only update parameters every num_accumulated_gradients optimization steps
        # use mean of gradients at every update
        if num_accumulated_gradients < update_every:
            for idx, param in enumerate(grad_params):
                accumulated_gradients[idx].append(param.grad.data.clone())
            num_accumulated_gradients += 1
        else:
            for idx, param in enumerate(grad_params):
                # rnn_initializers only have a gradient for first sequence of each episode
                if param.shape == (num_bees, hidden_size, 2):
                    stacked_grads = torch.stack(accumulated_gradients[idx])
                    has_valid_grad = stacked_grads.abs().sum(dim=(1, 2, 3)) > 0
                    if has_valid_grad.sum() > 0:
                        param.grad.data = stacked_grads[has_valid_grad].mean(dim=0)
                else:
                    param.grad.data = torch.stack(accumulated_gradients[idx]).mean(dim=0)
            optimizer.step()
            num_accumulated_gradients = 0
            accumulated_gradients = collections.defaultdict(list)

        # BPTT, detach gradients
        current_hs = list(map(lambda h: h.detach(), current_hs))
        current_cs = list(map(lambda c: c.detach(), current_cs))

    optimizer.zero_grad()
    last_loss = loss.cpu().data.item()
    loss = None

    return loss, last_loss, accumulated_gradients, num_accumulated_gradients, current_hs, current_cs


def subsample_data(df, sample_frac, random_start=False, random_end=False, truncate_every_n_steps=None):
    df = df.sample(frac=sample_frac).copy()
    df.sort_values('timestamp', inplace=True)

    if random_start or random_end:
        assert truncate_every_n_steps is not None

    start_idx = np.random.randint(len(df) - truncate_every_n_steps * 2) if random_start else 0
    end_idx = np.random.randint(start_idx + truncate_every_n_steps, len(df)) if random_end else len(df)
    df = df[start_idx:end_idx]

    df.reset_index(inplace=True, drop=False)

    df['dts'] = -1
    # recalculate time between events for sampled data
    for _, group in df.groupby('bee_id_0'):
        dts = group.timestamp.diff().apply(lambda dt: dt.total_seconds()) / (24 * 60)
        df.iloc[dts.index, -1] = dts.fillna(0).values

    return df


def train(df_full, dfs_by_bee, max_train_idx, sample_frac, num_epochs, lstm, optimizer,
          rnn_initializers, truncate_every_n_steps, update_every, num_bees, hidden_size,
          bee_ids, losses=None):
    one_day = np.timedelta64(1, 'D')

    current_hs = []
    current_cs = []

    num_accumulated_gradients = 0
    accumulated_gradients = collections.defaultdict(list)

    if losses is None:
        losses = []

    for epoch_idx in range(num_epochs):
        # sample epoch, fraction of temporal subset of data
        df = subsample_data(df_full[:max_train_idx], sample_frac=sample_frac)

        print('Epoch {} | from {} to {}'.format(epoch_idx, df.iloc[0].timestamp, df.iloc[-1].timestamp))

        if epoch_idx > 0:
            # TBPTT: detach gradients
            current_hs = list(map(lambda h: h.detach(), current_hs))
            current_cs = list(map(lambda c: c.detach(), current_cs))

        optimizer.zero_grad()
        loss = None

        # static learned initialized of rnn hidden state and context at beginning of episode
        hs, cs = [rnn_initializers[:, :, 0],
                  rnn_initializers[:, :, 1]]

        # we have to maintain a list of current hidden states because PyTorch can't backprop
        # if we modify them inplace
        current_hs = list(hs)
        current_cs = list(cs)

        for idx in range(len(df)):
            # BPTT, doesn't do anything for idx == 0 because loss is None at this time
            if (idx > 0) & (idx % truncate_every_n_steps == 0):
                loss, last_loss, accumulated_gradients, num_accumulated_gradients, current_hs, current_cs = optimization_step(
                    loss, accumulated_gradients, num_accumulated_gradients,
                    optimizer, current_hs, current_cs, update_every,
                    num_bees, hidden_size)

                losses.append(last_loss)

            row = df.iloc[idx, :]

            # current hidden state(s) from both interaction partners
            h = current_hs[row.bee_id_0][None, :]
            c = current_cs[row.bee_id_0][None, :]
            h_other = current_hs[row.bee_id_1][None, :]

            # time of event as a fraction of days in radians
            time = row.timestamp.time()
            time_days = (time.hour + time.minute / 60 + time.second / (60 * 60) + time.microsecond / (1e6 * 60 * 60)) / 24
            time_rad = time_days * 2 * np.pi

            # lstm features: time since last event from bee_0, absolute time representation, and
            # current hidden state from interaction partner
            features = np.stack((row['dts'], np.cos(time_rad), np.sin(time_rad)), axis=-1)
            features = torch.from_numpy(features[None, :].astype(np.float32))
            features = torch.cat((features, h_other), dim=-1)

            h, c = lstm(features, (h, c))

            current_hs[row.bee_id_0] = h[0]
            current_cs[row.bee_id_0] = c[0]

            df_online = dfs_by_bee[row.bee_id_0].loc[row['index']:]

            next_events = df_online[~df_online.bee_id_1.duplicated()]
            dts = (next_events.timestamp - row.timestamp).values / one_day

            ids = next_events.bee_id_1.values
            values = np.ones(len(bee_ids), dtype=np.float32) * np.nan
            values[ids] = dts

            targets = torch.from_numpy(values) * 24

            if not torch.any(torch.isfinite(targets)):
                continue

            mapped_embeddings = torch.stack(current_hs)

            # predicted interaction rate of current bee with all other bees
            rates = torch.nn.functional.softplus((mapped_embeddings @ mapped_embeddings[[row.bee_id_0]].T)) + 1e-8

            # negative log-likelihood of exponential distribution
            valid_targets = torch.isfinite(targets)
            neg_log_prob = -(rates[valid_targets].log() - rates[valid_targets] * targets[valid_targets, None])

            ## ignore likelihood to interact with yourself
            if np.isfinite(values).sum() > 0:
                # TODO: this may also mask nan computation errors
                neg_log_prob = neg_log_prob.mean()

                assert torch.isfinite(neg_log_prob)

                # accumulate loss for all time steps in current BPTT interval
                if loss is None:
                    loss = neg_log_prob
                else:
                    loss += neg_log_prob

        loss, last_loss, accumulated_gradients, num_accumulated_gradients, current_hs, current_cs = optimization_step(
            loss, accumulated_gradients, num_accumulated_gradients,
            optimizer, current_hs, current_cs, update_every,
            num_bees, hidden_size)

        losses.append(last_loss)
        print(np.mean(losses[-100:]))