train_result = []
# for dats in train_loader:
for dats, jid in zip(train_loader, train_loader.dataset.ids):
# A batch in the data loader
    info = {}
    # info["id"] = jid
    optimizer.zero_grad()
    if (config.model.alignn_layers) > 0:
        result = net([dats[0].to(device), dats[1].to(device)])
    else:
        result = net(dats[0].to(device))
    # info = {}
    info["target_out"] = []
    info["pred_out"] = []

    loss = 0
    if config.model.output_features is not None:
        # print('result',result)
        # print('dats[2]',dats[2])
        loss = criterion(
            result,
            dats[-1].to(device),
            # result, dats[2].to(device)
        )
        info["target_out"] = dats[-1].cpu().numpy().tolist()
        # info["target_out"] = dats[2].cpu().numpy().tolist()
        info["pred_out"] = (
            result.cpu().detach().numpy().tolist()
        )
    train_result.append(info)
    loss.backward()
    optimizer.step()
    # optimizer.zero_grad() #never
    running_loss += loss.item()
train_mean_loss = get_batch_errors(
    train_result
)