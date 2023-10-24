import torch
import torch.optim as optim
t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)
t_un = 0.1 * t_u
def model(t_u, w, b):
    return w*t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

'''def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs+1):
        if params.grad is not None:
            params.grad.zero_()
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():
            params -= learning_rate * params.grad
        print(f'Epoch:{epoch}  Loss:{float(loss)}')

#w = torch.ones(())
#b = torch.zeros(())
training_loop(n_epochs = 10, learning_rate = 1e-2, params = torch.tensor([1.0, 0.0], requires_grad = True), t_u = t_un, t_c = t_c)'''

'''params = torch.tensor([1.0, 0.0], requires_grad = True)
learning_rate = 1e-1
#optimizer = optim.SGD([params], lr = learning_rate)
optimizer = optim.Adam([params], lr = learning_rate)
t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch}  Loss:{float(loss)}')
        return

#training_loop(n_epochs = 10, optimizer = optimizer, params = params, t_u = t_un, t_c = t_c)'''

n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]
val_indices = shuffled_indices[-n_val:]

train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]
val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]

train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u

def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print(f'Epoch:{epoch}, Training loss:{train_loss.item():4f}, Validation loss{val_loss.item():4f}')
    return params

params = torch.tensor([1.0, 0.0],requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr = learning_rate)
training_loop(n_epochs = 10, optimizer = optimizer, params = params, train_t_u = train_t_un, val_t_u = val_t_un, train_t_c = train_t_c, val_t_c = val_t_c)