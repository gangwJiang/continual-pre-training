import json
import jsonlines
from transformers import PreTrainedTokenizer

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid=None, text_a=None, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                    input_ids=None,
                    input_mask=None,
                    segment_ids=None,

                    tokens_term_ids=None,
                    tokens_sentence_ids=None,

                    term_input_ids=None,
                    term_input_mask=None,
                    term_segment_ids=None,

                    sentence_input_ids=None,
                    sentence_input_mask=None,
                    sentence_segment_ids=None,

                    tokens_term_sentence_ids=None,
                    label_id=None,

                    masked_lm_labels = None,
                    masked_pos = None,
                    masked_weights = None,

                    position_ids=None,

                    valid_ids=None,
                    label_mask=None

                    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

        self.label_id = label_id

        self.masked_lm_labels = masked_lm_labels,
        self.masked_pos = masked_pos,
        self.masked_weights = masked_weights

        self.tokens_term_ids = tokens_term_ids
        self.tokens_sentence_ids = tokens_sentence_ids

        self.term_input_ids = term_input_ids
        self.term_input_mask = term_input_mask
        self.term_segment_ids = term_segment_ids

        self.sentence_input_ids = sentence_input_ids
        self.sentence_input_mask = sentence_input_mask
        self.sentence_segment_ids = sentence_segment_ids

        self.tokens_term_sentence_ids= tokens_term_sentence_ids

        self.position_ids = position_ids

        self.valid_ids = valid_ids
        self.label_mask = label_mask
        
        
class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json file for tasks in sentiment analysis."""
        if input_file.endswith(".json"):
            with open(input_file) as f:
                return json.load(f)
        elif input_file.endswith(".jsonl"):
            lines = []
            with open(input_file) as f:
                for item in jsonlines.Reader(f):
                    lines.append(item)
                return lines
        
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

        
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer: PreTrainedTokenizer, mode):
    """Loads a data file into a list of `InputBatch`s.""" #check later if we can merge this function with the SQuAD preprocessing
    
    label_map={}
    if mode == 'asc': # for pair
        label_map={'+': 0, 'positive': 0, '-': 1, 'negative': 1, 'neutral': 2}
    elif mode == 'nli':
        label_map={'neutral': 0, 'entailment': 1, 'contradiction': 2}
    elif mode == 'ae':
        label_map={'B': 0, 'I': 1, 'O': 2}
    features = []
    
    id2label = {}
    for (ex_index, example) in enumerate(examples):
        if example.text_b:
            tokenized = tokenizer.encode_plus((example.text_a, example.text_b), truncation=True, add_special_tokens=True, padding=True, return_token_type_ids=True, max_length=max_seq_length)
        else:
            if mode == 'ae':
                tokenized = tokenizer.encode_plus(example.text_a, truncation=True, add_special_tokens=True, padding=True, return_token_type_ids=True, is_split_into_words=True, max_length=max_seq_length)
            else:
                tokenized = tokenizer.encode_plus(example.text_a, truncation=True, add_special_tokens=True, padding=True, return_token_type_ids=True, max_length=max_seq_length)
           
        input_ids = tokenized["input_ids"]
        input_mask = tokenized["attention_mask"]
        segment_ids = tokenized["token_type_ids"]
        
        while len(input_ids) < max_seq_length:
            input_ids.append(tokenizer.pad_token_id)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        
        if mode in ["asc", "nli"]:
            label_id = label_map[example.label]
            id2label[label_id] = example.label
        elif mode in ["ae"]:
            label_id = [-1] * len(input_ids) #-1 is the index to ignore
            #truncate the label length if it exceeds the limit.
            lb = []
            for label in example.label:
                lb.append(label_map[label])
                id2label[label_map[label]] = label
            if len(lb) > max_seq_length - 2:
                lb = lb[0:(max_seq_length - 2)]
            label_id[1:len(lb)+1] = lb
        else:
            if example.label not in label_map.keys():
                label_map[example.label] = len(label_map) 
            label_id = label_map[example.label]
            id2label[label_id] = example.label

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id))
    
    return features, id2label
        # print("raw", example.text_a, example.text_b, example.label)
        # if mode!="ae":
        #     tokens_a = tokenizer.tokenize(example.text_a)
        # else: #only do subword tokenization.
        #     tokens_a, labels_a, example.idx_map= tokenizer.subword_tokenize([token.lower() for token in example.text_a], example.label )

        # tokens_b = None
        # if example.text_b:
        #     tokens_b = tokenizer.tokenize(example.text_b)
        # print("token", tokens_a, tokens_b)
        # if tokens_b:
        #     # Modifies `tokens_a` and `tokens_b` in place so that the total
        #     # length is less than the specified length.
        #     # Account for [CLS], [SEP], [SEP] with "- 3"
        #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        # else:
        #     # Account for [CLS] and [SEP] with "- 2"
        #     if len(tokens_a) > max_seq_length - 2:
        #         tokens_a = tokens_a[0:(max_seq_length - 2)]

        # tokens = []
        # segment_ids = []
        # tokens.append("[CLS]")
        # segment_ids.append(0)
        # for token in tokens_a:
        #     tokens.append(token)
        #     segment_ids.append(0)
        # tokens.append("[SEP]")
        # segment_ids.append(0)

        # if tokens_b:
        #     for token in tokens_b:
        #         tokens.append(token)
        #         segment_ids.append(1)
        #     tokens.append("[SEP]")
        #     segment_ids.append(1)
        # input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # # tokenizer.
        # # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        # input_mask = [1] * len(input_ids)

        # token_a has a max_length
        # if transformer_args.exp in ['3layer_aspect','2layer_aspect_transfer','2layer_aspect_dynamic']:
        #     term_position = tokens.index('[SEP]')-1
        #     while term_position < transformer_args.max_term_length: #[CLS],t,[SEP]
        #         input_ids.insert(term_position,0)
        #         input_mask.insert(term_position,0)
        #         segment_ids.insert(term_position,0)
        #         term_position+=1
        #     max_seq_length = max_seq_length
        
        # Zero-pad up to the sequence length.
        # while len(input_ids) < max_seq_length:
        #     input_ids.append(tokenizer.pad_token_id)
        #     input_mask.append(0)
        #     segment_ids.append(0)

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length

        # if mode!="ae":
        #     label_id = label_map[example.label]
        # else:
        #     label_id = [-1] * len(input_ids) #-1 is the index to ignore
        #     #truncate the label length if it exceeds the limit.
        #     lb=[label_map[label] for label in labels_a]
        #     if len(lb) > max_seq_length - 2:
        #         lb = lb[0:(max_seq_length - 2)]
        #     label_id[1:len(lb)+1] = lb

        # features.append(
        #         InputFeatures(
        #                 input_ids=input_ids,
        #                 input_mask=input_mask,
        #                 segment_ids=segment_ids,
        #                 label_id=label_id))