from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from data_loaders import GridNav_Dataset
from util.transforms import SelectChannelsTransform
from util.util import *
from models.VAE import *

#Select the directory using this
dataset_size = 120000
task_structures = ('rooms_unstructured_layout','maze') #('rooms_unstructured_layout','maze')
data_type = 'graph' #'graph'
data_dim = 27

use_gpu = True
plot_every = 10
epochs = 100
batch_size = 64
arch = 'big_gnn_sym_inv'#'fc'
latent_dim = 64

task_structures = '-'.join(task_structures)
dataset_directory = 'test'#f"ts={task_structures}-x={data_type}-s={dataset_size}-d={data_dim}"
run_name = 'test'#f"{dataset_directory}_arch={arch}-z={latent_dim}_b={batch_size}-e={epochs}"# 'test'#'multiroom10000x27_6cnn_z12_b64e1000' #CHANGE

writer = SummaryWriter('runs/' + run_name)
base_dir = str(Path(__file__).resolve().parent)
datasets_dir = base_dir + '/datasets/'

cifar_dir = datasets_dir + 'cifar10_data'
mnist_dir = datasets_dir #+ 'MNIST'

nav_dir = datasets_dir + dataset_directory
# model_config_filepath = base_dir + '/models/configs/GraphVAE.yaml'
# with open(model_config_filepath) as f:
#     model_config = yaml.safe_load(f)
#     model_config = dict2obj(model_config)

transform_data = True
if data_type == 'grid':
    layout_channels = (0,1)
elif data_type == 'gridworld':
    layout_channels = (0,)
elif data_type == 'graph':
    layout_channels = None
    transform_data = False

if transform_data is True:
    t = transforms.Compose([
        SelectChannelsTransform(*layout_channels),
        transforms.ToTensor(),])
else:
    t = None

# #Uncomment
train_data = GridNav_Dataset(
          nav_dir, train=True,
          transform = t)

test_data = GridNav_Dataset(
          nav_dir, train=False,
          transform = t)

# train_data = torchvision.datasets.MNIST(
#     mnist_dir, train=True, download=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         FlattenTransform(1,-1),
#         BinaryTransform(0.6),
#         ]))
# test_data = torchvision.datasets.MNIST(
#     mnist_dir, train=False,
#     transform=torchvision.transforms.Compose([
#         torchvision.transforms.Resize((img_size,img_size)),
#         torchvision.transforms.ToTensor(),
#         FlattenTransform(1, -1),
#         BinaryTransform(0.6),
#         ]))

if data_type == 'graph':
    train_loader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
else:
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# VAE setup

parser = create_VAE_argparser()
parser.print_help()

# Specify the hyperpameter choices
if data_type == 'graph':
    n_nodes = train_data[0][0].num_nodes()
    input_dim_flat = 1
    output_dim = (n_nodes - 1, 2)
    output_dim_flat = output_dim[0] * output_dim[1]
else:
    input_dim_flat = output_dim_flat = train_data[0][0].numel()
    output_dim = tuple(train_data[0][0].shape)

args_gnn = [  '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--data_dims', str(output_dim),
             '--epochs', str(epochs),
             '--learning_rate', '1e-4',
             '--cuda',
             'GNN',
             '--dec_layer_dims', f'{latent_dim}', '256', '1024', f'{output_dim_flat}',
             '--enc_layer_dims', f'{input_dim_flat}', '8', '1024', '256', f'{latent_dim}',
             '--enc_convolutions', '8',
             '--augmented_inputs',
              '--num_nodes', str(n_nodes),]

args_gnn_features = [  '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--data_dims', str(output_dim),
             '--epochs', str(epochs),
             '--learning_rate', '1e-4',
             '--cuda',
             'GNN',
             '--dec_layer_dims', f'{latent_dim}', '256', '1024', f'{output_dim_flat}',
             '--enc_layer_dims', f'{input_dim_flat}', '8', '1024', '256', f'{latent_dim}',
             '--enc_convolutions', '8',
             '--augmented_inputs',
              '--num_nodes', str(n_nodes),
            '--dec_output', json.dumps(decoder_config['output']),]

args_fc = [  '--gradient_type', 'pathwise',
             '--num_variational_samples', '1',
             '--data_distribution', 'Bernoulli',
             '--data_dims', str(output_dim),
             '--epochs', str(epochs),
             '--learning_rate', '1e-4',
             '--cuda',
             'FC',
             '--dec_layer_dims', f'{latent_dim}', '128', '256', f'{input_dim_flat}',
             '--enc_layer_dims', f'{input_dim_flat}', '256', '128', f'{latent_dim}',]

#Enc: 288 + 18432 + 36864 + 36864 + 73728 + 73728 + 6.4M+ 1M + 1M + 131k + 131k+ 1k ~ 9M
args_cnn_fc = ['CNN',
                '--dec_layer_dims', '12', '1024', '128,13,13', '64,27,27', '32,27,27', '1,27,27',
                '--dec_kernel_size', '3',
                 '--enc_layer_dims', '1,27,27', '32,27,27', '64,14,14', '64,14,14', '128,7,7', '128,7,7', '1024', '1024', '128', '12',
                 '--enc_kernel_size', '3',
                 '--gradient_type', 'pathwise',
                 '--num_variational_samples', '1',
                 '--data_distribution', 'Bernoulli',
                 '--epochs', str(epochs),
                 '--learning_rate', '1e-4',
                 '--cuda']

args_grid_cnn_fc = ['--gradient_type', 'pathwise',
                    '--num_variational_samples', '1',
                    '--data_distribution', 'Bernoulli',
                    '--epochs', str(epochs),
                    '--learning_rate', '1e-4',
                    '--cuda',
                    'CNN',
                    '--dec_layer_dims', '12', '1024', '256,5,5', '256,7,7', '64,9,9', '64,11,11', '16,13,13', '2,13,13',
                    '--dec_kernel_size', '3',
                    '--enc_layer_dims', '2,13,13', '16,13,13', '64,11,11', '64,9,9', '256,7,7', '256,5,5', '1024',
                    '1024', '128', '12',
                    '--enc_kernel_size', '3',]

if arch == '6cnn' or arch == '5cnn':
    if data_type == 'gridworld':
        args = args_cnn_fc# CHANGE
    elif data_type == 'grid':
        args = args_grid_cnn_fc
elif arch == 'fc':
    args = args_fc
elif 'gnn' in arch:
    args = args_gnn_features

args = parser.parse_args(args)# CHANGE
args.batch_size = batch_size
#args.seed = 10111201

if use_gpu == True:
    args.cuda = args.cuda and torch.cuda.is_available()
else:
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

# Seed all random number generators for reproducibility of the runs
seed_everything(args.seed)

# Initialise the model and the Adam (SGD) optimiser
model = VAE(args).to(args.device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

model, optimizer, model_state_best_training_elbo, optim_state_best_training_elbo, early_t = fit_model(model, optimizer,
                                                      train_data, args,
                                                      test_data=test_data, latent_eval_freq=plot_every, tensorboard=writer)
if early_t: run_name = run_name + '_early_t'
save_file = 'checkpoints/' + run_name + '.pt'
print(f"Saving to {save_file}")
save_state(args, model, optimizer, save_file, [model_state_best_training_elbo], [optim_state_best_training_elbo])

writer.close()

print("Done")