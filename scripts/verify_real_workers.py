"""Measure actual CPU encoders and indexes without requiring an LLM provider."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import config
import numpy as np
import torch
from media_tools import ffmpeg_path
from evaluation.memory import process_memory
from vector_validation import unit_vector
import semantic_search
import clap_search

report=dict(evidence='real_cpu_workers_and_indexes_no_llm',musical_quality_evaluated=False,
    provider_evaluated=False,revisions=config.MODEL_REVISIONS,checks={})
for name,worker,query in [('lyrics',semantic_search.search_lyrics,'returning home'),('audio',clap_search.search_audio,'piano')]:
    samples=[]
    for repeat in range(4):
        started=time.perf_counter()
        results=worker(positive=[query],negative=[],top_k=3)
        samples.append(dict(seconds=time.perf_counter()-started,song_ids=[row[0] for row in results],**process_memory()))
    report['checks'][name]=dict(status='completed',samples=samples,first_sample_includes_model_load=True,
        cold_disk_cache_verified=False,warm_samples=3)
source=json.loads((ROOT/'validation/autonomous/live-ingestion.json').read_text(encoding='utf-8'))
media=next((ROOT/'tmp/autonomous-live').glob('*/ingestion/media/'+source['song']['audio_file']))
raw=subprocess.run([ffmpeg_path(),'-nostdin','-v','error','-i',str(media),'-t','2','-f','f32le','-ac','1','-ar','48000','pipe:1'],check=True,capture_output=True).stdout
resources=clap_search._load_resources()
with torch.no_grad():
    inputs=resources['processor'](audio=np.frombuffer(raw,dtype='<f4').copy(),sampling_rate=48000,return_tensors='pt').to(clap_search.DEVICE)
    encoded=resources['model'].get_audio_features(**inputs)
    vector=unit_vector(encoded.squeeze().cpu().numpy(),512)
report['checks']['native_audio']=dict(status='completed',dimensions=len(vector),norm=float(np.linalg.norm(vector)),
    audio_source='downloaded yt-dlp test video',**process_memory())
target=ROOT/'validation/autonomous/real-workers.json'
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
