"""Real CPU smoke checks for Whisper and Nomic; no relevance labels are invented."""
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
import numpy as np
from vector_validation import unit_vector

report=dict(evidence='real_local_models_cpu',musical_quality_evaluated=False,checks={})
try:
    import whisper
    started=time.perf_counter()
    print('Loading Whisper base',flush=True)
    model=whisper.load_model('base',device='cpu',download_root=str(ROOT/'.tools/whisper'))
    result=model.transcribe(np.zeros(32000,dtype=np.float32),fp16=False,condition_on_previous_text=False)
    report['checks']['whisper']=dict(status='completed',model='base',input='two seconds of generated silence',
        elapsed_seconds=time.perf_counter()-started,returned_text=result['text'],transcript_accuracy_evaluated=False)
except Exception as exc:
    report['checks']['whisper']=dict(status='failed',error=str(exc)); traceback.print_exc()
try:
    print('Loading Nomic',flush=True)
    started=time.perf_counter()
    import semantic_search
    resources=semantic_search._load_resources()
    vector=unit_vector(resources['embedder'].encode('search_document: Returning home to my family',normalize_embeddings=True)[:256],256)
    report['checks']['nomic']=dict(status='completed',dimensions=len(vector),norm=float(np.linalg.norm(vector)),
        elapsed_seconds=time.perf_counter()-started)
except Exception as exc:
    report['checks']['nomic']=dict(status='failed',error=str(exc)); traceback.print_exc()
report['downloaded_revisions']={str(path.relative_to(ROOT)):path.read_text().strip()
    for path in (ROOT/'.tools/huggingface/hub').glob('models--*/refs/main')}
target=ROOT/'validation/autonomous/optional-models.json'
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
raise SystemExit(0 if all(row['status']=='completed' for row in report['checks'].values()) else 1)
