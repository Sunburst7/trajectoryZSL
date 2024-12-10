from yacs.config import CfgNode as CN

_C = CN()
_C.random_seed = 2024
_C.root_project_path = "/data2/hh/workspace/trajectoryZSL"

_C.dataset = CN()
_C.dataset.root_data_path = "/data2/hh/workspace/data/ais"
_C.dataset.num_class = 14
_C.dataset.num_feature = 8
_C.dataset.seq_len = 40
_C.dataset.ratio = 0.7
_C.dataset.is_gzsl = False
_C.dataset.seen_class =  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]
_C.dataset.unseen_class = [10]

_C.stft = CN()
_C.stft.n_fft = 128
_C.stft.window_length = 128
_C.stft.hop_length = 16
_C.stft.window_function = "Hamming"

_C.model = CN()
_C.model.eta_cent = 5e-2
_C.model.eta_cls = 1
_C.model.eta_margin = 0.1
_C.model.learning_rate = 0.001409257213017734
_C.model.learning_rate_cent = 0.5

_C.model.margin = 24
_C.model.num_epoch = 70
_C.model.batch_size = 16
_C.model.wd = 0.1
_C.model.dropout = 0.1
_C.model.devices = [0, 1, 2, 3] # [i for i in range(torch.cuda.device_count())]

_C.model.task_name = "classification"
_C.model.seq_len = _C.dataset.seq_len
_C.model.pred_len = 0
_C.model.output_attention = False
_C.model.embed = 'fixed'
_C.model.freq = 'h'
_C.model.activation = 'gelu'
_C.model.factor = 1
_C.model.d_model = 124
_C.model.n_heads = 8
_C.model.d_ff = 108 # dimension of model
_C.model.e_layers = 6
_C.model.enc_in = _C.dataset.num_feature # encoder input size
_C.model.num_class = _C.dataset.num_class
_C.model.d_center = _C.model.d_model * _C.model.enc_in

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`