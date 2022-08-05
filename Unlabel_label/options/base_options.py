import argparse
import os
from utils import utils
import torch

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument("--comment", default='', type=str,
                            help="The comment")
        parser.add_argument('--output_dir', type=str, default="../../ablation")
        parser.add_argument('--cache_dir', type=str, default=None)
        parser.add_argument("--continual_train", action='store_true',
                            help="Whether restore from the last checkpoint, is nochenckpoints, start from scartch")
        parser.add_argument("--block_size", default=512, type=int,
                            help="ptional input sequence length after tokenization."
		"The training dataset will be truncated in block of this size for training."
		"Default to the model max input length for single sentence inputs (take into account special tokens)")
       

        # aux dataset (unlabeled) parameters
        parser.add_argument( "--aux_task_types", default=None, type=str, 
                            help="The input training data file(s). Number of files specified must match number of aux-task-names")   
        parser.add_argument("--aux_task_names", default=None, type=str, help="The names of the auxiliary tasks")
        parser.add_argument("--overwrite_cache", action='store_true',
                            help="overwrite the cache of the aux features")
        parser.add_argument("--lazy_dataset", action='store_true',
                            help="tokenize dataset when initial or load")
        parser.add_argument("--line_by_line", action='store_true',
                            help="tokenize dataset when initial or load")
        parser.add_argument("--mlm_probability", type=float, default=0.15, 
                            help="Ratio of tokens to mask for masked language modeling loss")
            
        # downstream dataset (labeled) parameters
        parser.add_argument("--train_tasks", default=None, type=str, 
                            help="The input training data file(s)")        
        parser.add_argument("--pri_task", default="seq_cla", type=str, 
                            help="The task of primary task(s)")    
        parser.add_argument("--task_order", default=None, type=str, 
                            help="The order of training task(s)")    
        parser.add_argument("--task_mix", action='store_true',
                            help="Whether to mix up the data in the same domain") 
        parser.add_argument("--max_seq_len", type=int, default=256, 
                            help="max lenght of token sequence")
           
        
        # model parameters
        parser.add_argument("--model_type", default='roberta-base', type=str, 
                            help="Model type selected in the list")
        parser.add_argument("--model_name_or_path", default='roberta-base', type=str, 
                            help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--config_name", default='roberta-base', type=str, 
                    help="Path to pre-trained model or shortcut name selected in the list: ")
        parser.add_argument("--tokenizer_name", default='roberta-base', type=str, 
            help="Path to pre-trained model or shortcut name selected in the list: ")
        
        # optimize parameters
        parser.add_argument("--per_gpu_train_batch_size", default=2, type=int,
                            help="Batch size per GPU/CPU for training.")
        parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                            help="Batch size per GPU/CPU for evaluation.")
        parser.add_argument("--betas", type=str, default="(0.9,0.98)",
                            help="The initial beta for Adam.")
        parser.add_argument("--learning_rate", default=3e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--base_wd", type=float, default=0.01)
        parser.add_argument("--weight_decay", default=0.0, type=float,
                            help="Weight deay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                            help="Epsilon for Adam optimizer.")    
        parser.add_argument("--warmup_frac", type=float, default=0.06)
        parser.add_argument("--max_grad_norm", default=1.0, type=float,
                            help="Max gradient norm.")
        parser.add_argument("--num_train_epochs", default=1.0, type=float, 
                            help="Total number of training epochs to perform.")
        # parser.add_argument("--max_steps", default=-1, type=int,
        #                     help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--eval_every", type=int, default=300,
                            help="Frequency with which to evaluate the model")
        
        
        
        
        parser.add_argument("--do_train", action='store_true',
                            help="Whether to run training.")
        parser.add_argument("--do_eval", action='store_true',
                            help="Whether to run eval on the dev set.")
        parser.add_argument("--evaluate_during_training", type=bool, default=False,
                            help="Rul evaluation during training at each logging step.")
        parser.add_argument("--do_lower_case", action='store_true',
                            help="Set this flag if you are using an uncased model.")

        parser.add_argument("--adapter_transformer_layers", default=2, type=int,
                            help="The transformer layers of adapter.")
        parser.add_argument("--adapter_size", default=128, type=int,
                            help="The hidden size of adapter.")
        parser.add_argument("--adapter_list", default="0,11,23", type=str,
                            help="The layer where add an adapter")
        parser.add_argument("--adapter_skip_layers", default=6, type=int,
                            help="The skip_layers of adapter according to bert layers")
        parser.add_argument('--meta_adapter_model', type=str, help='the pretrained adapter model')

       
        
        parser.add_argument('--logging_steps', type=int, default=10,
                            help="How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)")
        parser.add_argument('--save_steps', type=int, default=1000,
                            help="Save checkpoint every X updates steps.")
        parser.add_argument('--eval_steps', type=int, default=None,
                            help="eval every X updates steps.")
        parser.add_argument('--max_save_checkpoints', type=int, default=500,
                            help="The max amounts of checkpoint saving. Bigger than it will delete the former checkpoints")
        parser.add_argument("--eval_all_checkpoints", action='store_true',
                            help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
        parser.add_argument("--no_cuda", action='store_true',
                            help="Avoid using CUDA when available")
        parser.add_argument('--overwrite_output_dir', action='store_true',
                            help="Overwrite the content of the output directory")
        parser.add_argument('--seed', type=int, default=42,
                            help="random seed for initialization")

        parser.add_argument('--fp16', action='store_true',
                            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        parser.add_argument('--fp16_opt_level', type=str, default='O1',
                            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html")
        parser.add_argument("--local_rank", type=int, default=-1,
                            help="For distributed training: local_rank")
        parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
        parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
        parser.add_argument('--negative_sample', type=int, default=0, help='how many negative samples to select')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            self.parser = self.initialize(parser)
        # save and return the parser
        args = parser.parse_args()        
        return args

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        file_name = os.path.join(opt.output_dir, 'train_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        name_prefix = 'aux-' + str(opt.aux_task_names) + '_' +  'down-' + str(opt.train_tasks) + '_'  + 'maxlen-' + str(
                        opt.max_seq_len) + '_' + 'batch-' + str(
                        opt.per_gpu_train_batch_size) + '_' + 'lr-' + str(opt.learning_rate) + '_' + 'warmup-' + str(
                        opt.warmup_steps) + '_' + 'epoch-' + str(opt.num_train_epochs) + '_' + str(opt.comment)
        opt.model_name = name_prefix
        opt.output_dir = os.path.join(opt.output_dir, opt.model_name)
        utils.mkdirs(opt.output_dir)
        
        self.print_options(opt)
        
        opt.train_tasks = opt.train_tasks.split(":") if opt.train_tasks else []
        opt.aux_task_names = opt.aux_task_names.split(":") if opt.aux_task_names else []
        opt.aux_task_types = opt.aux_task_types.split(":") if opt.aux_task_types else []
        opt.task_order = opt.task_order.split(":") # example: order = ["a1,d1,d2,"]
        if len(opt.task_order) != len(opt.train_tasks)+len(opt.aux_task_names):
            raise ValueError("Unmatched order with the provided dataset")
        
        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt