import collections
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
    loss = None

    return loss, accumulated_gradients, num_accumulated_gradients, current_hs, current_cs