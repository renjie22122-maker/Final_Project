"""Check local dependencies and expected weights without running inference."""
import argparse
import importlib.util
import os
from pathlib import Path

WEIGHTS = {
    'cyclegan': ['cyclegan/m2r/200_net_G.pth', 'cyclegan/r2m/200_net_G.pth'],
    'cut': ['cut/m2r/400_net_G.pth', 'cut/r2m/400_net_G.pth'],
    'unsb': ['unsb/m2r/400_net_G.pth', 'unsb/r2m/400_net_G.pth'],
    'cyclegan-turbo': ['cyclegan_turbo/model_25001.pkl', 'cyclegan_turbo/fixed_prompt_a.txt', 'cyclegan_turbo/fixed_prompt_b.txt'],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['all', *WEIGHTS], default='all')
    args = parser.parse_args()
    selected = WEIGHTS if args.model == 'all' else {args.model: WEIGHTS[args.model]}
    root = Path(os.environ.get('MONET_MODEL_ROOT', str(Path(__file__).parent / 'models')))
    missing = 0
    modules = ['flask', 'flask_cors', 'PIL', 'numpy', 'torch', 'torchvision', 'dominate', 'packaging']
    if 'unsb' in selected:
        modules += ['lpips']
    if 'cyclegan-turbo' in selected:
        modules += ['diffusers', 'transformers', 'peft', 'huggingface_hub']
    for name in modules:
        found = importlib.util.find_spec(name) is not None
        print(('OK      ' if found else 'MISSING ') + name)
        missing += not found
    try:
        import torch
        print('CUDA available:', torch.cuda.is_available())
        if 'cyclegan-turbo' in selected and not torch.cuda.is_available():
            print('MISSING CUDA: vendored Turbo code requires CUDA')
            missing += 1
    except ImportError:
        pass
    for paths in selected.values():
        for relative in paths:
            found = (root / relative).is_file()
            print(('OK      ' if found else 'MISSING ') + str(root / relative))
            missing += not found
    print('This checks presence only; it does not validate checkpoint compatibility or inference quality.')
    return 1 if missing else 0

if __name__ == '__main__':
    raise SystemExit(main())
