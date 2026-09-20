import os
import sys
import uuid
import torch
import traceback  # 顶部导入
from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

# === 路径配置 ===
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
MODEL_ROOT = "models"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
conversion_history = []

DIRECTION_MAP = {"m2r": "a2b", "r2m": "b2a"}

import importlib.util

def load_module_safely(module_name, module_path):
    """动态加载指定模块路径，返回模块对象"""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# === CycleGAN-Turbo 推理 ===
def transform_image_turbo(image_path, direction):
    import os
    import sys
    import uuid
    import torch
    from PIL import Image
    from torchvision import transforms
    from transformers import AutoTokenizer, CLIPTextModel

    sys.path = [p for p in sys.path if "src" not in p]  # 去掉所有 src 路径
    for k in list(sys.modules.keys()):
        if k.startswith("models") or k.startswith("data") or k.startswith("options"):
            del sys.modules[k]

    sys.path.insert(0, os.path.abspath("src/cyclegan_turbo/src"))  # 或 cut/unsb


    from cyclegan_turbo import (
        CycleGAN_Turbo,
        VAE_encode,
        VAE_decode,
        initialize_unet,
        initialize_vae,
        make_1step_sched,
    )

    model_direction = DIRECTION_MAP.get(direction)
    if model_direction is None:
        raise ValueError(f"Unsupported direction: {direction}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 文本编码 ===
    model_dir = os.path.join(MODEL_ROOT, "cyclegan_turbo")
    prompt_file = os.path.join(model_dir, "fixed_prompt_a.txt" if model_direction == "a2b" else "fixed_prompt_b.txt")
    with open(prompt_file, "r", encoding="utf-8") as f:
        caption = f.read().strip()

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").to(device).eval()
    token_ids = tokenizer(caption, return_tensors="pt", padding="max_length", truncation=True,
                          max_length=tokenizer.model_max_length)["input_ids"].to(device)
    text_emb = text_encoder(token_ids)[0]

    # === 加载 checkpoint ===
    checkpoint_path = os.path.join(model_dir, "model_25001.pkl")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # === 构建 UNet 和 VAE ===
    unet = initialize_unet(rank=checkpoint["rank_unet"]).to(device)
    vae_a2b = initialize_vae(rank=checkpoint["rank_vae"]).to(device)
    vae_b2a = initialize_vae(rank=checkpoint["rank_vae"]).to(device)

    # === 手动注入 LoRA 权重 ===
    for n, p in unet.named_parameters():
        if "lora" in n:
            if "default_encoder" in n:
                key = n.replace(".default_encoder.weight", ".weight")
                p.data.copy_(checkpoint["sd_encoder"][key])
            elif "default_decoder" in n:
                key = n.replace(".default_decoder.weight", ".weight")
                p.data.copy_(checkpoint["sd_decoder"][key])
            elif "default_others" in n:
                key = n.replace(".default_others.weight", ".weight")
                p.data.copy_(checkpoint["sd_other"][key])
    unet.set_adapter(["default_encoder", "default_decoder", "default_others"])
    unet.eval()

    # === 清洗 VAE 权重前缀 ===
    def clean_vae_keys(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("vae."):
                new_key = k[len("vae."):]
            elif k.startswith("vae_a2b."):
                new_key = k[len("vae_a2b."):]
            elif k.startswith("vae_b2a."):
                new_key = k[len("vae_b2a."):]
            else:
                new_key = k
            new_state_dict[new_key] = v
        return new_state_dict

    vae_a2b.load_state_dict(clean_vae_keys(checkpoint["sd_vae_enc"]), strict=False)
    vae_b2a.load_state_dict(clean_vae_keys(checkpoint["sd_vae_dec"]), strict=False)
    vae_a2b.eval()
    vae_b2a.eval()

    # === 编码与解码器封装 ===
    vae_enc = VAE_encode(vae_a2b, vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a)

    # === Scheduler 和 timestep ===
    scheduler = make_1step_sched()
    timesteps = torch.tensor([scheduler.config.num_train_timesteps - 1], device=device).long()

    # === 图像预处理 ===
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    # === 推理 ===
    with torch.no_grad():
        output_tensor = CycleGAN_Turbo.forward_with_networks(
            image_tensor, model_direction,
            vae_enc, unet, vae_dec,
            scheduler, timesteps, text_emb
        )

    # === 输出图像保存 ===
    output_tensor = (output_tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    output_image = transforms.ToPILImage()(output_tensor.cpu())
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    output_image.save(output_path)
    return output_filename


# === CycleGAN 推理 ===
def transform_image_cyclegan(image_path, direction):
    import shutil
    import argparse
    from PIL import Image

    # 动态处理路径与模块冲突
    sys.path = [p for p in sys.path if "src" not in p]  
    for k in list(sys.modules.keys()):
        if k.startswith("models") or k.startswith("data") or k.startswith("options"):
            del sys.modules[k]

    sys.path.insert(0, os.path.abspath("src/cyclegan"))  


    from options.test_options import TestOptions
    from data import create_dataset
    from models import create_model
    from util.util import tensor2im

    temp_dataroot = "temp_test_dir"
    os.makedirs(temp_dataroot, exist_ok=True)
    temp_image_path = os.path.join(temp_dataroot, os.path.basename(image_path))
    shutil.copy(image_path, temp_image_path)

    # direction -> name
    model_name = direction  
    checkpoint_dir = os.path.join("models", "cyclegan")

    opt_obj = TestOptions()
    parser = argparse.ArgumentParser()
    parser = opt_obj.initialize(parser)
    opt = parser.parse_args(args=[
        "--dataroot", temp_dataroot,
        "--name", model_name,
        "--checkpoints_dir", checkpoint_dir,
        "--model", "test",
        "--dataset_mode", "single",
        "--direction", "AtoB" if direction == "m2r" else "BtoA",
        "--no_dropout",
        "--no_flip",
        "--serial_batches",
        "--num_threads", "0",
        "--batch_size", "1",
        "--load_size", "256",
        "--crop_size", "256",
        "--epoch", "200",
        "--phase", "test",
        "--num_test", "1"
    ])

    # 必要补充字段
    opt.isTrain = False
    opt.model_suffix = ''
    opt.gpu_ids = [0] if torch.cuda.is_available() else []

    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    for data in dataset:
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        image_tensor = visuals.get("fake_B") or visuals.get("fake")
        break

    output_image = Image.fromarray(tensor2im(image_tensor))
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    output_image.save(output_path)

    shutil.rmtree(temp_dataroot)
    return output_filename

# === CUT 推理 ===
def transform_image_cut(image_path, direction):
    import sys
    import os
    import uuid
    import torch
    from PIL import Image

    # === 添加 CUT 模型源码路径到 sys.path ===
    sys.path = [p for p in sys.path if "src" not in p]  
    for k in list(sys.modules.keys()):
        if k.startswith("models") or k.startswith("data") or k.startswith("options"):
            del sys.modules[k]

    sys.path.insert(0, os.path.abspath("src/cut"))  


    from options.test_options import TestOptions
    from models import create_model
    from data.base_dataset import get_transform
    from util.util import tensor2im

    # === 配置参数 ===
    opt = TestOptions().parse()
    opt.name = direction                      
    opt.model = "cut"
    opt.CUT_mode = "CUT"
    opt.no_dropout = True
    opt.direction = "AtoB" if direction == "m2r" else "BtoA"
    opt.isTrain = False
    opt.load_size = 256
    opt.crop_size = 256
    opt.preprocess = "resize_and_crop"
    opt.serial_batches = True
    opt.no_flip = True
    opt.num_threads = 0
    opt.batch_size = 1
    opt.checkpoints_dir = os.path.abspath("models/cut")
    opt.epoch = "400"           # 对应 400_net_G.pth
    opt.verbose = False         # 禁用 options 输出

    # === 创建并加载模型 ===
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # === 图像预处理 ===
    image = Image.open(image_path).convert("RGB")
    transform = get_transform(opt)
    image_tensor = transform(image).unsqueeze(0)

    # === 推理过程 ===
    with torch.no_grad():
        data = {
            "A": image_tensor,
            "B": image_tensor,              # 必须提供 B，否则 KeyError
            "A_paths": image_path,
            "B_paths": image_path
        }
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        output_tensor = visuals["fake_B"]

    # === 保存图像 ===
    output_image = Image.fromarray(tensor2im(output_tensor))
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join("outputs", output_filename)
    output_image.save(output_path)
    return output_filename

# === UNSB 推理 ===
def transform_image_unsb(image_path, direction):
    import os, sys, uuid, shutil
    import torch
    from PIL import Image
    from torchvision import transforms
    import contextlib

    sys.path = [p for p in sys.path if "src" not in p]  # 去掉所有 src 路径
    for k in list(sys.modules.keys()):
        if k.startswith("models") or k.startswith("data") or k.startswith("options"):
            del sys.modules[k]

    sys.path.insert(0, os.path.abspath("src/unsb"))  # 或 cut/unsb


    from options.test_options import TestOptions
    from models import create_model
    from data import create_dataset
    from util.util import tensor2im

    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

    # 构造 temp_test_dir/testA 和 testB 结构
    temp_root = "temp_test_dir"
    testA_dir = os.path.join(temp_root, "testA")
    testB_dir = os.path.join(temp_root, "testB")
    os.makedirs(testA_dir, exist_ok=True)
    os.makedirs(testB_dir, exist_ok=True)

    # 拷贝真实图像至 testA
    testA_img_path = os.path.join(testA_dir, os.path.basename(image_path))
    shutil.copy(image_path, testA_img_path)

    # 构造 dummy 图像至 testB
    dummy_img = Image.new("RGB", (256, 256), (0, 0, 0))
    dummy_img_path = os.path.join(testB_dir, "dummy.jpg")
    dummy_img.save(dummy_img_path)

    # 解析参数并静默打印
    with suppress_stdout():
        opt = TestOptions().parse()

    # 手动修改参数
    opt.dataroot = temp_root
    opt.name = direction
    opt.model = "sb"
    opt.mode = "sb"
    opt.checkpoints_dir = "models/unsb"
    opt.no_dropout = True
    opt.direction = "AtoB"  # 始终为 AtoB！
    opt.isTrain = False
    opt.load_size = 256
    opt.crop_size = 256
    opt.preprocess = "resize_and_crop"
    opt.serial_batches = True
    opt.no_flip = True
    opt.num_threads = 0
    opt.batch_size = 1
    opt.epoch = "400"
    opt.phase = "test"
    opt.num_test = 1
    opt.eval = True

    opt.gpu_ids = [0] if torch.cuda.is_available() else []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    model = create_model(opt)
    model.data_dependent_initialize(next(iter(dataset)), next(iter(dataset2)))
    model.setup(opt)
    model.parallelize()
    model.eval()

    for data, data2 in zip(dataset, dataset2):
        model.set_input(data, data2)
        model.test()
        visuals = model.get_current_visuals()

        # 获取输出图像 tensor
        output_tensor = None
        for key in ["fake_5", "fake_4", "fake_3", "fake_2", "fake_1"]:
            if key in visuals:
                output_tensor = visuals[key]
                print(f"✅ 使用模型输出 key: {key}")
                break

        if output_tensor is None:
            raise RuntimeError(f"❌ 模型输出无有效图像，visuals keys 为: {list(visuals.keys())}")

        break

    # 保存结果
    output_image = Image.fromarray(tensor2im(output_tensor))
    output_filename = f"{uuid.uuid4().hex}.png"
    output_path = os.path.join("outputs", output_filename)
    output_image.save(output_path)

    shutil.rmtree(temp_root)
    return output_filename

# === Flask 路由 ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("file")
    filenames = []
    for file in files:
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)
        filenames.append(file.filename)
    return jsonify({"filenames": filenames})

@app.route("/apply_style", methods=["POST"])
def apply_style():
    data = request.json
    filename = data.get("filename")
    direction = data.get("direction", "m2r")
    model_name = data.get("model", "cyclegan-turbo")

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(input_path):
        return jsonify({"error": "File not found"}), 404

    try:
        if model_name == "cyclegan-turbo":
            output_filename = transform_image_turbo(input_path, direction)
        elif model_name == "cyclegan":
            output_filename = transform_image_cyclegan(input_path, direction)
        elif model_name == "cut":
            output_filename = transform_image_cut(input_path, direction)
        elif model_name == "unsb":
            output_filename = transform_image_unsb(input_path, direction)
        else:
            return jsonify({"error": f"Unknown model '{model_name}'"}), 400

    except Exception as e:
        # 捕获完整 traceback 信息
        error_msg = traceback.format_exc()

        # 打印到服务器控制台
        print("Error during transformation:\n", error_msg)

        # 写入 debug 日志
        with open("debug_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"\n[ERROR] model={model_name}, direction={direction}, filename={filename}\n")
            log_file.write(error_msg + "\n")

        # 返回错误信息给前端
        return jsonify({"error": f"{str(e)}", "trace": error_msg}), 500

    # 正常返回
    conversion_history.append({
        "input": filename,
        "output": output_filename,
        "direction": direction,
        "model": model_name
    })
    return jsonify({"output_filename": output_filename})

@app.route("/uploads/<filename>")
def upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/outputs/<filename>")
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/history")
def history():
    return jsonify({"history": conversion_history})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
