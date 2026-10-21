"""Exercise native adapters in an isolated catalog; never alter production data."""
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['HF_HOME']=str(ROOT/'.tools/huggingface')
from catalog import CatalogManager
from config import OUTPUT_DIR
from ingestion import IngestionService

target=ROOT/'validation/autonomous'
target.mkdir(parents=True,exist_ok=True)
root=ROOT/'tmp/autonomous-live'/uuid4().hex
root.mkdir(parents=True)
manager=CatalogManager(root/'catalog',OUTPUT_DIR)
manager.current()
service=IngestionService(manager,root/'ingestion')
report=dict(evidence='native_youtube_pipeline_isolated_catalog',url='https://www.youtube.com/watch?v=DTi8wZ1a1TA',
            capabilities=service.status(),real_models=True,musical_quality_evaluated=False)
try:
    job=service.enqueue(report['url'],dict(instrumental=True))
    end=time.monotonic()+900
    prior=None
    while time.monotonic()<end:
        job=service.store.get(job['id'])
        if job['stage']!=prior: print(job['stage'],flush=True); prior=job['stage']
        if job['status'] in {'completed','failed','interrupted'}: break
        time.sleep(1)
    report['job']=job
    report['status']='completed' if job['status']=='completed' else 'failed'
    if job['status']=='completed':
        snapshot=manager.current()
        report['catalog_version']=snapshot.version
        report['song']=snapshot.get_song_by_id(job['result']['song_id'])
        report['lyrics_vectors']=snapshot.idx_lyrics.ntotal
        report['audio_vectors']=snapshot.idx_audio.ntotal
finally:
    service.close()
    (target/'live-ingestion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False),flush=True)
raise SystemExit(0 if report.get('status')=='completed' else 1)
