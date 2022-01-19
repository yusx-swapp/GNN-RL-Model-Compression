

# load target neural network
device = torch.device(args.device)

net = load_model(args.model,args.data_root)
net.to(device)