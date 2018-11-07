# Seed project for experimenting with Tensorflow model

Supports:

- Load json file as model configuration

- Auto save and resume from previous checkpoint

## Training arguments

- `--reset`: ignore last checkpoint
- `--load`: load from specified checkpoint instead of last checkpoint
- `--transfer`: use `transfer_load()` defined in model

## JSON configurations

- `model`: path to import model class
- `dataset`: path to import dataset
- `verbose`: output debug information
- `batch_size`
- `max_epoch_num`
- `colocate_gradients_with_ops`

### Optimization

- `optimizer`
- `learning_rate`
- `learning_rate_start_decay_epoch`
- `learning_rate_decay_steps`
- `learning_rate_decay_rate`
- `max_gradient_norm`