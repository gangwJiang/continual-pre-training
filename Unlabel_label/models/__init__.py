import os
from utils import utils
from transformers import (
	AutoConfig,
	AutoModelWithLMHead,
	AutoTokenizer,
)

from models.question_answer import ExtenedQuestionAnswering
from models.seq_classification import ExtenedSeqClassification
from models.token_classification import ExtenedTokenClassification


def create_model(args):
    # if args.model_type not in ["bert", "roberta", "distilbert", "camembert"]:
    #     raise ValueError(
    #         "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
    #         "flag (masked language modeling)."
    #     )

    if args.continual_train:
        sorted_checkpoints = utils._sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --continual_train but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]
            print('Used Should Continue and model found is : ', args.model_name_or_path)
            
        
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    # Setting up tokenizer and pre-trained model
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        model = AutoModelWithLMHead.from_config(config)



    return model, tokenizer



def create_pri_model(args, num_labels):
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, output_hidden_states=True, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, output_hidden_states=True, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )
    
    config.num_labels = num_labels
    
    pri_model = None
    if args.pri_task == "token_cla":
        pri_model = ExtenedTokenClassification(config)
    elif args.pri_task == "seq_cla":
        pri_model = ExtenedSeqClassification(config)
    elif args.pri_task == "qa":
        pri_model = ExtenedQuestionAnswering(config)
        
    return pri_model