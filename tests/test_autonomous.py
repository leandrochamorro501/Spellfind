"""Independent modality maps, storage quotas and offline recovery regressions."""
from pathlib import Path
import json
import os
import shutil
import time
import unittest
from unittest.mock import patch
from uuid import uuid4
import zipfile

import numpy as np
import faiss
from catalog import CatalogManager, FILES, read_bundle
from ingestion import JobStore, ingest_payload
from maintenance import cleanup, backup, restore, verify_backup, safe_path, StorageLimit
from resources import search_resources, use_snapshot
from test_block5 import bundle, URL


class AutonomousTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[1] / 'tmp/autonomous-tests'
        self.root = self.base / uuid4().hex
        self.root.mkdir(parents=True)
        self.legacy = self.root / 'legacy'
        bundle(self.legacy)
        self.manager = CatalogManager(self.root/'catalog', self.legacy)
        self.initial = self.manager.current()
        self.ingest = self.root/'ingestion'
        self.ingest.mkdir()

    def tearDown(self):
        target = self.root.resolve()
        if target.parent != self.base.resolve(): raise ValueError('Unsafe test cleanup')
        shutil.rmtree(target)

    def instrumental(self, **extra):
        job = uuid4().hex
        metadata = dict(title='Instrumental control', artist='Control', lyrics='', source_url=URL, **extra)
        return self.manager.append_song(metadata, None, np.ones(512), job_id=job)

    def test_audio_only_and_later_lyric_song_use_independent_positions(self):
        first, sid = self.instrumental()
        self.assertEqual((first.idx_lyrics.ntotal, first.idx_audio.ntotal), (2,3))
        self.assertNotIn(sid, first.mapping_for('lyrics').values())
        last, sid2 = self.manager.append_song(dict(title='Lyrics', artist='Control', lyrics='Coming home'),
            np.ones(256), np.ones(512), job_id=uuid4().hex)
        self.assertEqual(last.mapping_for('lyrics')['2'], sid2)
        self.assertEqual(last.mapping_for('audio')['2'], sid)
        with use_snapshot(last):
            self.assertEqual(search_resources({'embedder': object()})['song_id_map'], last.mapping_for('lyrics'))
            self.assertEqual(search_resources({'model': object()})['song_id_map'], last.mapping_for('audio'))
        loaded = CatalogManager(self.manager.root, self.legacy).current()
        self.assertEqual(loaded.mapping_for('lyrics'), last.mapping_for('lyrics'))

    def test_audio_only_cannot_claim_a_lyric_vector_or_missing_audio(self):
        with self.assertRaises(ValueError):
            self.manager.append_song(dict(title='Control', lyrics=''), np.ones(256), np.ones(512), job_id=uuid4().hex)
        with self.assertRaises(ValueError):
            self.manager.append_song(dict(title='Control', lyrics=''), None, np.zeros(512), job_id=uuid4().hex)
        self.assertEqual(self.manager.current().version, self.initial.version)

    def test_catalog_with_no_lyric_vectors_can_later_add_lyrics(self):
        source = self.root/'audio-only'
        shutil.copytree(self.legacy,source)
        import pandas as pd
        frame = pd.read_csv(source/FILES[0]); frame['lyrics']=''
        frame.to_csv(source/FILES[0],index=False)
        mapping = json.loads((source/FILES[1]).read_text())
        (source/FILES[1]).write_text(json.dumps(dict(schema_version=2,catalog=mapping,lyrics={},audio=mapping)))
        faiss.write_index(faiss.IndexFlatIP(256),str(source/FILES[2]))
        empty = self.manager.publish_directory(source)
        self.assertEqual(empty.idx_lyrics.ntotal,0)
        snapshot,sid = self.manager.append_song(dict(title='Lyrics',lyrics='Returning home'),
            np.ones(256),np.ones(512),job_id=uuid4().hex)
        self.assertEqual(snapshot.mapping_for('lyrics'),{'0':sid})

    def test_independent_map_cannot_reference_an_unknown_song(self):
        source = self.root/'invalid-map'
        shutil.copytree(self.legacy,source)
        mapping = json.loads((source/FILES[1]).read_text())
        (source/FILES[1]).write_text(json.dumps(dict(schema_version=2,catalog=mapping,lyrics={'0':999,'1':2},audio=mapping)))
        with self.assertRaises(ValueError): self.manager.publish_directory(source)
        self.assertEqual(self.manager.current().version,self.initial.version)

    def test_instrumental_payload_requires_boolean(self):
        self.assertTrue(ingest_payload(dict(url=URL,instrumental=True))[1]['instrumental'])
        for value in ('true', 1, None):
            with self.assertRaises(ValueError): ingest_payload(dict(url=URL, instrumental=value))

    def test_storage_admission_and_catalog_quota_preserve_live_version(self):
        with patch.dict(os.environ, {'SPELLFIND_CATALOG_QUOTA_BYTES':'1'}), self.assertRaises(StorageLimit):
            self.instrumental()
        self.assertEqual(self.manager.current().version, self.initial.version)
        store = JobStore(self.ingest)
        with patch.dict(os.environ, {'SPELLFIND_INGEST_QUOTA_BYTES':'1'}), self.assertRaises(StorageLimit):
            store.enqueue(URL,{})

    def test_backup_restore_keeps_catalog_media_and_hashes(self):
        name = uuid4().hex+'.mp3'
        (self.ingest/'media').mkdir()
        (self.ingest/'media'/name).write_bytes(b'ID3-control')
        snapshot, sid = self.instrumental(audio_file=name)
        archive = self.root/'backup.zip'
        backup(self.manager,self.ingest,archive)
        self.assertIn('catalog/registry.sqlite3',verify_backup(archive)['sha256'])
        restored = self.root/'restored'
        result = restore(archive,restored)
        self.assertEqual(result['catalog_version'],snapshot.version)
        self.assertEqual((restored/'ingestion/media'/name).read_bytes(),b'ID3-control')
        with self.assertRaises(FileExistsError): restore(archive,restored)

    def test_backup_refuses_active_work_and_does_not_leave_partial_archive(self):
        JobStore(self.ingest).enqueue(URL,{})
        with self.assertRaises(RuntimeError): backup(self.manager,self.ingest,self.root/'backup.zip')
        self.assertFalse((self.root/'backup.zip.partial').exists())

    def test_corrupt_backup_is_not_restored(self):
        archive = self.root/'backup.zip'
        backup(self.manager,self.ingest,archive)
        with zipfile.ZipFile(archive,'a') as zipped: zipped.writestr('ingestion/extra.txt','unexpected')
        with self.assertRaises(ValueError): restore(archive,self.root/'restored')
        self.assertFalse((self.root/'restored').exists())

    def test_cleanup_preview_and_quarantine_keep_current_media(self):
        name = uuid4().hex+'.mp3'
        media = self.ingest/'media'; media.mkdir()
        (media/name).write_bytes(b'ID3-control')
        snapshot, sid = self.instrumental(audio_file=name)
        old = time.time()-40*86400
        with self.manager.transaction() as db: db.execute('UPDATE history SET created=? WHERE version=?',(old,self.initial.version))
        previous = self.manager.root/'versions'/self.initial.version
        os.utime(previous,(old,old))
        orphan = media/(uuid4().hex+'.mp3'); orphan.write_bytes(b'unused'); os.utime(orphan,(old,old))
        preview = cleanup(self.manager,self.ingest,keep_versions=1)
        self.assertIn(self.initial.version,preview['expires_audio_versions'])
        self.assertTrue(previous.exists()); self.assertTrue(orphan.exists())
        result = cleanup(self.manager,self.ingest,apply=True,keep_versions=1)
        self.assertTrue(result['quarantine'])
        self.assertTrue(all(Path(entry['quarantine']).stat().st_mtime > time.time()-60 for entry in result['quarantine']))
        self.assertTrue((media/name).is_file())
        self.assertFalse(orphan.exists())
        self.assertEqual(self.manager.current().version,snapshot.version)
        with self.assertRaises(ValueError): self.manager.at_version(self.initial.version)

    def test_cleanup_aborts_when_a_retained_generation_is_corrupt(self):
        (self.manager.root/'versions'/self.initial.version/FILES[0]).write_text('broken')
        with self.assertRaises(ValueError): cleanup(self.manager,self.ingest,apply=True)

    def test_maintenance_paths_cannot_escape_managed_roots(self):
        for path in ('../outside','..','nested/../../outside'):
            with self.assertRaises(ValueError): safe_path(self.root,path)


if __name__ == '__main__': unittest.main()
