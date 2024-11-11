test_result = []
for dats, jid in zip(test_loader, test_loader.dataset.ids):
    # for dats in test_loader:
    info = {}
    info["id"] = jid
    optimizer.zero_grad()
    # print('dats[0]',dats[0])
    # print('test_loader',test_loader)
    # print('test_loader.dataset.ids',test_loader.dataset.ids)
    # result = net([dats[0].to(device), dats[1].to(device)])
    if (config.model.alignn_layers) > 0:
        result = net([dats[0].to(device), dats[1].to(device)])
    else:
        result = net(dats[0].to(device))
    loss = 0
    if (
        config.model.output_features is not None
        and not classification
    ):
        # print('result',result)
        # print('dats[2]',dats[2])
        loss = criterion(
            result, dats[-1].to(device)
        )
        info["target_out"] = dats[-1].cpu().numpy().tolist()
        info["pred_out"] = (
            result.cpu().detach().numpy().tolist()
        )

    test_result.append(info)