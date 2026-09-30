这个问题问到了 DDPM 公式体系的一个关键点。直接回答：**`α_t = 1 - β_t` 不是"推导出来的物理定律"，而是一个人为的定义（记号约定）**。它背后的动机是"方差保持"这个设计选择。下面拆开讲。

---

## 1. β_t 才是"本体"，α_t 只是它的补数

DDPM 的前向过程被定义成一条马尔可夫链，**每一步加一点高斯噪声**：

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(x_t;\ \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I\big)
$$

其中 **β_t 就是"第 t 步加噪声的方差"**，也就是噪声调度（noise schedule）。它是我们**主动选择**的一个序列，通常从很小（1e-4）单调增到 0.02（对应代码 [DiffsFormer.py:27-31](model/DiffsFormer.py#L27-L31) 的 `torch.linspace(1e-4, 0.02, T)`）。

用重参数化写出来就是：

$$
x_t = \sqrt{1-\beta_t}\, x_{t-1} + \sqrt{\beta_t}\,\varepsilon,\qquad \varepsilon \sim \mathcal{N}(0, I)
$$

看这两个系数：**信号项系数是 $\sqrt{1-\beta_t}$，噪声项系数是 $\sqrt{\beta_t}$**。

于是定义一个记号 $\alpha_t := 1 - \beta_t$，上式就变成更清爽的形式：

$$
x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1-\alpha_t}\,\varepsilon
$$

所以 `alphas = 1.0 - betas`（[DiffsFormer.py:384](model/DiffsFormer.py#L384)）**纯粹是为了让符号变短、好写**。α_t 的物理含义是"**单步信号保留系数**"——一步之后原信号还有多少比例被保留。

---

## 2. 为什么系数要取 `sqrt(1-β_t)` 和 `sqrt(β_t)`？——方差保持

这是问题的核心。**为什么信号系数偏要用 $\sqrt{1-\beta_t}$，噪声系数偏要用 $\sqrt{\beta_t}$？** 因为这样两个系数的**平方和恒等于 1**：

$$
\big(\sqrt{1-\beta_t}\big)^2 + \big(\sqrt{\beta_t}\big)^2 = (1-\beta_t) + \beta_t = 1
$$

假设 $x_{t-1}$ 每个坐标的方差是 1（归一化后），且与噪声 ε 独立，那么：

$$
\text{Var}(x_t) = (\sqrt{1-\beta_t})^2 \cdot \text{Var}(x_{t-1}) + (\sqrt{\beta_t})^2 \cdot \text{Var}(\varepsilon) = (1-\beta_t)\cdot 1 + \beta_t\cdot 1 = 1
$$

**每一步方差都保持为 1**。这就是所谓的 **方差保持（Variance-Preserving, VP）** 扩散。好处是：数据从 t=0 的"干净信号"到 t=T 的"纯噪声"，方差一路上不爆炸也不塌缩，数值上非常稳定。（另有"方差爆炸 Variance-Exploding"的变体，DDPM 走的是 VP 这条路。）

---

## 3. 这个定义的最大便利：累积积 ᾱ_t 给出闭式解

`α_t = 1 - β_t` 这个约定真正"值钱"的地方，在于它让**多步累积**变得极简洁。定义累积积：

$$
\bar\alpha_t = \prod_{s=1}^{t} \alpha_s
$$

因为前向过程是高斯且系数是线性的，可以**把所有中间步一次性边缘化掉**，直接写出从 $x_0$ 到 $x_t$ 的闭式：

$$
q(x_t \mid x_0) = \mathcal{N}\big(x_t;\ \sqrt{\bar\alpha_t}\, x_0,\ (1-\bar\alpha_t) I\big)
$$

即：

$$
x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon
$$

这正是代码里 `diffuse` 的那一行（[DiffsFormer.py:149](model/DiffsFormer.py#L149)）。这个闭式解是 DDPM 的命门：

1. **训练**时不用一步步走，任意 t 直接采样 $x_t$，算 MSE 损失 $\|\varepsilon - \varepsilon_\theta(x_t,t)\|^2$ 即可；
2. 系数平方和仍是 $\bar\alpha_t + (1-\bar\alpha_t) = 1$，方差保持性质在累积后依然成立；
3. 因为 $\beta_t \in (0,1)$，所以 $\alpha_t \in (0,1)$，于是 $\bar\alpha_t$ 随 t 单调降到接近 0 —— 保证 $t=T$ 时 $x_T \approx$ 纯噪声，这是"扩散"能成立的边界条件。

---

## 4. 对应到代码（完整链条）

[DiffsFormer.py:382-389](model/DiffsFormer.py#L382-L389) 的这四行，就是上面全部公式的落地：

```python
self.register_buffer("betas", make_beta_schedule(total_timesteps, beta_start, beta_end))  # β_t：噪声方差
self.register_buffer("alphas", 1.0 - self.betas)                    # α_t = 1 - β_t
self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))  # ᾱ_t = ∏ α_s
self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))
```

---

**一句话总结**：`α_t = 1 - β_t` 是一个**记号定义**——β_t 是每步加噪声的方差，α_t 是"信号保留系数"，定义成补数的目的是让前向转移写成 $\sqrt{\alpha_t}$ 和 $\sqrt{1-\alpha_t}$ 两个平方和为 1 的系数，从而**保持方差、并让累积积 $\bar\alpha_t$ 直接给出从 $x_0$ 到 $x_t$ 的闭式解**。