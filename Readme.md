### base_test

the basic implement of pre training model

1. text classification on distilbert 
    
    text_class_dbert.py
    
2. 


### Unlabeled-labeled

```
CUDA_VISIBLE_DEVICES=9 python run_mlm_mix.py --aux_task_types mlm:mlm --aux_task_names bio:cs --train_tasks chemprot:citation_intent --task_order d0:a0:a1:d1  &> /data2/home/gangwei/project/ablation/9.txt
```