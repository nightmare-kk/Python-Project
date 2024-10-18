# 基于BERT微调的微博评论情感分析
## 简介
* 使用Bert-base-Chinese，对微博评论数据集进行情感分类，分别是[积极/消极]。

## Requirements
~~~shell
pip install -r requirements.txt
~~~
## Dataset
使用[WeiboSenti100k](https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k)数据集，该数据集包含在./dataset。
该数据集包含10万条中国微博帖子，每条帖子都被标记为正面或负面。

## Usage
* ### Train
    **Run the Code** 
    ~~~shell
        python train.py
    ~~~
  
    **Train Args**
    * output_dir: 模型保存路径
    * warmup_ratio: warmup比例
    * learning_rate: 学习率
    * per_device_train_batch_size: 每个设备的训练批次大小
    * per_device_eval_batch_size: 每个设备的评估批次大小
    * num_train_epochs: 训练轮数
    * logging_dir: 日志保存路径
    * logging_steps: 日志保存步数
    * evaluation_strategy: 评估策略
    * save_strategy: 保存策略
    * save_total_limit: 保存模型的最大数量
    * weight_decay: 权重衰减
    * load_best_model_at_end: 在训练结束时加载最佳模型
    * metric_for_best_model: 用于选择最佳模型的指标

* ### Inference
  ~~~shell
      python inference.py -s "There input your sentence." # Inference for single sentence.
  ~~~
  ~~~shell
      python inference.py -i True # Continuous inference.
  ~~~

## Related Docs
* *<a href="https://huggingface.co/docs">HuggingFace Docs</a>*
* *<a href="https://pytorch.org/docs/stable/index.html">Pytorch Docs</a>*
