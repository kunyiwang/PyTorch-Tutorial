from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter('logs')

    for i in range(100):
        writer.add_scalar('y=2x', 2*i, i)

    writer.close()

# Following is command to visualize the log file:
# tensorboard --logdir=logs