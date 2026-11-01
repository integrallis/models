"""Remote GPU-only bounded training; qualification data is never opened here."""
import argparse
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common import HERE, digest, verify_snapshot


def main():
    p = argparse.ArgumentParser()
    for name in ['snapshot', 'source', 'data', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    # Fail before loading weights if the environment drifted from the frozen recipe.
    for pin in (HERE/'requirements-training.in').read_text().splitlines():
        name, expected = pin.split('==')
        if importlib.metadata.version(name) != expected:
            raise ValueError('training package version mismatch: '+name)
    policy = json.loads((HERE/'finetune-policy.json').read_text())
    splits = json.loads((HERE/'finetune-splits.json').read_text())
    for name in ['train.jsonl', 'development.jsonl']:
        if digest(args.data/name) != splits['files'][name]['sha256']:
            raise ValueError('training/development identity mismatch: '+name)
    revision = subprocess.check_output(['git','-C',str(args.source),'rev-parse','HEAD'],text=True).strip()
    if revision != policy['publisherCommit']:
        raise ValueError('wrong publisher source revision')
    if subprocess.check_output(['git','-C',str(args.source),'status','--porcelain'],text=True).strip():
        raise ValueError('publisher checkout must be clean')
    base_manifest = verify_snapshot(args.snapshot)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    sys.path.insert(0,str(args.source))
    import torch
    from gliner2 import AutoExtractor
    from gliner2.training.trainer import ExtractorTrainer, TrainingConfig
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('requires a BF16-capable GPU; no CPU training fallback')
    torch.set_num_threads(4)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('training output must be empty; never select an old best checkpoint')
    args.output.mkdir(parents=True,exist_ok=True)
    checkpoint_dir = args.output/'checkpoints'
    model = AutoExtractor.from_pretrained(str(args.snapshot),local_files_only=True,
                                          attention_backend='standard',map_location='cpu')
    class TrainingBudgetReached(Exception):
        pass
    class BoundedTrainer(ExtractorTrainer):
        deadline = float('inf')
        def _optimizer_step(self):
            if time.monotonic() >= self.deadline:
                raise TrainingBudgetReached()
            return super()._optimizer_step()
    config = TrainingConfig(output_dir=str(checkpoint_dir), experiment_name=policy['candidate'],
                            **policy['training'])
    compute_metrics=None
    if policy.get('developmentDiagnostics'):
        from development_metrics import measure_development, admission_margin
        development_path=args.data/'development-cases.jsonl'
        if digest(development_path)!=splits['files']['development-cases.jsonl']['sha256']:
            raise ValueError('development diagnostic data identity mismatch')
        development=[json.loads(line) for line in development_path.read_text().splitlines()]
        def compute_metrics(trained,dataset):
            metrics=measure_development(trained,development,policy['training']['bf16'])
            metrics['dev_admission_margin']=admission_margin(metrics,policy['quality'])
            print(json.dumps(dict(stage='development',metrics=metrics)),flush=True)
            return metrics
    trainer = BoundedTrainer(model=model,config=config,compute_metrics=compute_metrics)
    trainer.deadline = time.monotonic()+policy['maximumTrainingSeconds']
    started = time.monotonic(); status='completed'
    try:
        trainer.train(train_data=str(args.data/'train.jsonl'),eval_data=str(args.data/'development.jsonl'))
    except TrainingBudgetReached:
        status='training-time-limit'
        trainer.model.save_pretrained(str(checkpoint_dir/'budget-stop'))
        # The cutoff is not a selected checkpoint. Evaluate it on development data only.
        trainer._evaluate(trainer._prepare_data(str(args.data/'development.jsonl'),is_train=False))
    best = checkpoint_dir/'best'
    selected_file=best/('adapter_model.safetensors' if config.use_lora else 'model.safetensors')
    if not selected_file.is_file() or not math.isfinite(trainer.best_metric):
        raise RuntimeError('no valid development-selected checkpoint')
    record = dict(candidate=policy['candidate'],status=status,steps=trainer.global_step,
        trainingSeconds=time.monotonic()-started,selectionMetric=config.metric_for_best,
        bestSelectionMetric=trainer.best_metric,
        bestDevelopmentLoss=next(row['eval_loss'] for row in trainer.eval_metrics_history if row[config.metric_for_best]==trainer.best_metric),
        trainableParameters=sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        trainingHistory=trainer.train_metrics_history,
        developmentHistory=trainer.eval_metrics_history,baseSnapshot=base_manifest,
        trainingPolicy=policy,splitManifest=splits,gpu=torch.cuda.get_device_name(0),
        torchVersion=torch.__version__,cudaVersion=torch.version.cuda,
        packages={name:importlib.metadata.version(name) for name in ['transformers','tokenizers','peft','safetensors']},
        selectedCheckpointSha256=digest(selected_file),catalogQualified=False)
    trainer.model.cpu()
    del trainer,model
    gc.collect();torch.cuda.empty_cache()
    # Reload the development-selected checkpoint; never export the last training step accidentally.
    if config.use_lora:
        base = AutoExtractor.from_pretrained(str(args.snapshot),local_files_only=True,
                                             attention_backend='standard',map_location='cpu')
        selected = PeftModel.from_pretrained(base,str(best))
        merged = selected.merge_and_unload().float().eval()
    else:
        merged = AutoExtractor.from_pretrained(str(best),local_files_only=True,
                                               attention_backend='standard',map_location='cpu').float().eval()
    state = {name:tensor.detach().cpu().contiguous() for name,tensor in merged.state_dict().items()}
    with safe_open(str(args.snapshot/'model.safetensors'),framework='pt',device='cpu') as original:
        if set(state) != set(original.keys()):
            raise ValueError('merged tensor inventory differs from base')
        for name,tensor in state.items():
            if list(tensor.shape) != original.get_slice(name).get_shape() or tensor.dtype != torch.float32:
                raise ValueError('merged shape/dtype mismatch: '+name)
            if not torch.isfinite(tensor).all():
                raise ValueError('nonfinite merged tensor: '+name)
    output = args.output/'snapshot';output.mkdir(exist_ok=True)
    for entry in base_manifest['files']:
        if entry['path'] == 'model.safetensors': continue
        dest=output/entry['path'];dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(args.snapshot/entry['path'],dest)
    save_file(state,str(output/'model.safetensors'))
    manifest=dict(base_manifest,id=policy['candidate'],sourceUri='unpublished-training-output',
                  revision=digest(output/'model.safetensors'),derivedFrom=base_manifest['revision'])
    manifest['files']=[dict(entry,sha256=digest(output/entry['path']),sizeBytes=(output/entry['path']).stat().st_size)
                       for entry in base_manifest['files']]
    (args.output/'snapshot.json').write_text(json.dumps(manifest,indent=2)+'\n')
    record['mergedSnapshot']=manifest
    (args.output/'training-record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(candidate=policy['candidate'],steps=record['steps'],status=status,
                         bestDevelopmentLoss=record['bestDevelopmentLoss'],mergedWeightSha256=manifest['revision'])))


if __name__ == '__main__':
    main()
