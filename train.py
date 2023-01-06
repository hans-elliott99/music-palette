import torch
import torch.nn as nn
import models
import utils

if __name__=="__main__":
    X, Y, s = utils.load_and_split_data("./audio_mel", n_samples=100)
    X_train, X_val, X_test = X
    Y_train, Y_val, Y_test = Y
    s_train, s_val, s_test = s
    s, specs, pals = utils.prep_next_song(s_train, X_train, Y_train, song_id=s_train[0])
    n_samples_in_song = len(s)

    X = torch.tensor(specs[0]).unsqueeze(0) #batch_dim
    mod2 = models.ConvSequenceRNN(X.shape)





    # mod = models.ConvRNN(X_shape=specs[0].shape)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(mod.parameters(), lr=0.01)

    # for _ in range(10):
    #     optimizer.zero_grad(set_to_none=True)

    #     X = torch.tensor(specs[0])
    #     y = torch.tensor(pals[0], dtype=torch.float32)
    #     logits = mod(X)

    #     logits_stretched = logits.view(logits.size(0), -1)
    #     truth_stretched  = y.view(y.size(0), -1)

    #     loss = criterion(logits_stretched, truth_stretched)
    #     acc = utils.rgb_accuracy(logits, y, 100)
    #     print(loss.item())
    #     print(acc)

    #     loss.backward()
    #     optimizer.step()