"""Isolated browser fixture: real queue/catalog/FFmpeg, controlled AI."""
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import app
from catalog import CatalogManager
from config import OUTPUT_DIR
from ingestion import IngestionService
from media_tools import ffmpeg_path

root = ROOT/'tmp/autonomous-preview'/uuid4().hex
root.mkdir(parents=True)
manager = CatalogManager(root/'catalog',OUTPUT_DIR)
manager.current()


class Backend:
    def capabilities(self): return dict(ready=True,tools={'FFmpeg':True},model_weights_verified=False,missing=[])
    def run(self,job,work,progress):
        for stage, percent in [('downloading',10),('transcribing',40),('embedding_audio',80)]:
            progress(stage,percent); time.sleep(2)
        audio = work/'control.mp3'
        subprocess.run([ffmpeg_path(),'-nostdin','-v','error','-f','lavfi','-i','sine=frequency=440:duration=2',str(audio)],check=True)
        instrumental = job['metadata'].get('instrumental') is True
        return (dict(title='CONTROL instrumental añadido',artist='CONTROL local',tag='control',year='',
            lyrics='' if instrumental else 'Returning home in a controlled example',
            lyrics_source='none_user_declared_instrumental' if instrumental else 'control',
            instrumental_source='user_declared_not_acoustically_verified'),
            None if instrumental else np.ones(256), np.ones(512), audio)


def worker(mode):
    def search(**kwargs):
        from resources import _snapshot
        snapshot = _snapshot.get() or manager.current()
        return [(sid, .9-i*.001) for i,sid in enumerate(reversed(list(snapshot.mapping_for(mode).values())))][:kwargs.get('top_k',10)]
    return search


app._import_router=lambda:SimpleNamespace(invoke=lambda _:dict(lyric_positive=[],lyric_negative=[],clap_positive=['piano'],clap_negative=[]))
app._import_semantic_search=lambda:worker('lyrics')
app._import_clap_search=lambda:worker('audio')
service=IngestionService(manager,root/'ingestion',Backend())
application=app.create_app({'INGEST_ENABLED':True,'INGEST_DIR':str(root/'ingestion')},search_engine=app.SearchEngine(manager),ingest_service=service)
application.jinja_env.auto_reload=True


@application.after_request
def banner(response):
    if response.mimetype=='text/html':
        response.set_data(response.get_data(as_text=True).replace('<main',
            '<p style="background:#fff1b8;color:#332600;padding:12px;text-align:center">CONTROL LOCAL: cola y audio reales; IA simulada. Catálogo aislado.</p><main',1))
    return response


print('PREVIEW_ROOT='+str(root),flush=True)
try: application.run(host='127.0.0.1',port=5075,debug=False,use_reloader=False)
finally: service.close()
