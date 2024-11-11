val_result = []
# for dats in val_loader:
for dats, jid in zip(val_loader, val_loader.dataset.ids):
    info = {}
    # info["id"] = jid
    optimizer.zero_grad()
    # result = net([dats[0].to(device), dats[1].to(device)])
    if (config.model.alignn_layers) > 0:
        result = net([dats[0].to(device), dats[1].to(device)])
    else:
        result = net(dats[0].to(device))
    # info = {}
    info["target_out"] = []
    info["pred_out"] = []
    loss = 0
    if config.model.output_features is not None:
        loss = criterion(
            result, dats[-1].to(device)
        )
        info["target_out"] = dats[-1].cpu().numpy().tolist()
        info["pred_out"] = (
            result.cpu().detach().numpy().tolist()
        )

    val_result.append(info)
    val_loss += loss.item()

val_mean_loss = get_batch_errors(
    val_result
)