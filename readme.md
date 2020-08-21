**Status:** Maintenance (expect bug fixes and minor updates)

Welcome to Spinning Up in Deep RL! 
==================================

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!


Citing Spinning Up
------------------

If you reference or use Spinning Up in your research, please cite:

```
@article{SpinningUp2018,
    author = {Achiam, Joshua},
    title = {{Spinning Up in Deep Reinforcement Learning}},
    year = {2018}
}
```

## GSAC 运行命令

1. 直接运行

```
python -m spinup.run gsac_pytorch 
    --env HalfCheetah-v3 
    --exp_name gsac_Walker2d 
    --hid [400,300] 
    --epochs 750 
    --max_beta_pi 0.0 --max_beta_q 0.0 --max_bias_q 0.0 
    --seed 123 
    --device cuda:0 
```

命令解释:

- `gsac_pytorch` 表示用哪一个rl算法；
- `--env HalfCheetah-v3` 表示用哪个环境；
- `--exp_name gsac_Walker2d` 表示实验数据和网络模型保存的文件夹名字，该文件夹将出现在 `spinningup/data/` 目录下面；
- `--hid [400, 300]` 表示隐藏层为 $400 \times 300$ 的全连接层；
- `--epochs 750` 表示训练轮次，换算下来是 $10^6$ 的样本数；
- `--max_beta_pi 0.0 --max_beta_q 0.0 --max_bias_q 0.0` 表示一系列正则项的超参数，这里设置为0表示不加正则项;
    如果要设置 `flow` 有关的正则项的超参数，可以添加 `--beta_sh 1.0`。
    这里直接对应算法函数的参数表：

    ```python
    def gsac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
            steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
            polyak=0.995, pi_lr=1.0, lr=1e-3, batch_size=100, start_steps=10000, 
            update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
            logger_kwargs=dict(), save_freq=1, 
            device='cuda', expand_batch=100, 
            start_beta_pi=1.0, beta_pi_velocity=0.1, max_beta_pi=1.0,
            start_beta_q =0.0, beta_q_velocity =0.01, max_beta_q =1.0,
            start_bias_q =0.0, bias_q_velocity =0.1, max_bias_q =10.0, 
            warm_steps=0, reward_scale=1.0, kernel='energy', noise='gaussian',
            beta_sh = 1.0, eta=100, sh_p=2, blur_loss=10, blur_constraint=1, 
            scaling=0.95, backend="tensorized"):
    ```

- `--seed 123` 表示种子数；
- `--device cuda:0` 表示使用第一块GPU训练。

2. 挂载到后台的命令模板

```
nohup python -m spinup.run gsac_pytorch 
    --env HalfCheetah-v3 
    --exp_name gsac_Walker2d 
    --hid [400,300] 
    --epochs 750 
    --max_beta_pi 0.0 --max_beta_q 0.0 --max_bias_q 0.0 
    --seed 123 
    --device cuda:0 
> gsac_HalfCheetah_s123.log 2>&1 &
```

命令解释：挂在到后台，将 **本来会在shell窗口输出的内容** 转而输出到文件 `gsac_HalfCheetah_s123.log` 中。