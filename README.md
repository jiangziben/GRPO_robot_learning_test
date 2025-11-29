# GRPO-robot

### Environment:
```
conda create -n grpo-robot -c conda-forge python=3.7 pybox2d
conda activate grpo-robot
pip install numpy<1.19.5 gym<=0.25.2 torch matplotlib pygame seaborn tqdm 
```

### Run:
```
# 训练
python ./ppo_train_cartpole.py

# 将训练好的策略模型可视化
python ./强化学习过程可视化-cartpole.py
```

### 一些有用的参考

- [动手学强化学习](https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95)
- [RL可视化](https://github.com/pybox2d/pybox2d?tab=readme-ov-file#)
- [gym的使用](https://cloud.tencent.com/developer/article/2387660)
- [gym讲解](https://blog.csdn.net/qq_58718853/article/details/142137851)

