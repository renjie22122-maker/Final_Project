# 训练权重与数据

原项目提供的权重下载地址：[Google Drive 模型压缩包](https://drive.google.com/file/d/1WkVsXojtnuFdLrzccfye1eWWCdMn_1kp/view?usp=sharing)。该地址沿用已有 `models/anouncement.txt`；本次未验证匿名下载权限。压缩包约 1.39 GB，解压约 1.51 GB。下载不可用时请通过仓库维护者取得权重，不要把基础模型权重当作本项目已训练检查点。

解压后应满足以下路径（不要额外嵌套一层 `models/models`）：

```text
models/
  cyclegan/
    m2r/200_net_G.pth
    r2m/200_net_G.pth
  cut/
    m2r/400_net_G.pth
    r2m/400_net_G.pth
  unsb/
    m2r/400_net_G.pth
    r2m/400_net_G.pth
  cyclegan_turbo/
    model_25001.pkl
    fixed_prompt_a.txt
    fixed_prompt_b.txt
```

`m2r` 表示莫奈到照片，`r2m` 表示照片到莫奈。前述 G 文件为 Web 推理所需生成器；原训练归档中的 D/E/F、日志和配置可另行保留用于训练复现。Turbo 的两个 prompt 文件应使用训练归档的原文件，不能随意替换。应用还需取得 `stabilityai/sd-turbo` 的 tokenizer、text encoder、UNet 和 VAE；UNSB 现有实现会初始化 LPIPS/VGG 与 DINO，首次执行可能联网下载。

`MONET_MODEL_ROOT` 环境变量可指定外部模型目录。权重由 PyTorch 加载，仅使用可信来源文件。Git 忽略 `.pth`、`.pt`、`.pkl`、`.bin` 等大文件；模型和数据的使用条件分别以其来源为准。

训练数据不包含在本次补全中。根目录的十张展示图片来自毕业设计结果章节，只用于展示输入与输出；`dataset(sample_input_and_output)` 是早期提交的示例集合，不能据此推断完整训练/测试划分。
